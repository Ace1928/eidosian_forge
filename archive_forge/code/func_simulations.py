import datetime as dt
import gzip
import logging
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from yapf.yapflib.yapf_api import FormatCode
import statsmodels.api as sm
def simulations(sim_type, save=False):
    rs = np.random.RandomState(seed)
    remaining = NUM_SIM
    results = defaultdict(list)
    start = dt.datetime.now()
    while remaining > 0:
        this_iter = min(remaining, MAX_SIM_SIZE)
        remaining -= this_iter
        if sim_type == 'normal':
            dist = rs.standard_normal
        else:
            dist = rs.standard_exponential
        rvs = dist((MAX_SIZE, this_iter))
        sample_sizes = [ss for ss in SAMPLE_SIZES if ss >= MIN_SAMPLE_SIZE[sim_type]]
        for ss in sample_sizes:
            sample = rvs[:ss]
            mu = sample.mean(0)
            if sim_type == 'normal':
                std = sample.std(0, ddof=1)
                z = (sample - mu) / std
                cdf_fn = stats.norm.cdf
            else:
                z = sample / mu
                cdf_fn = stats.expon.cdf
            z = np.sort(z, axis=0)
            nobs = ss
            cdf = cdf_fn(z)
            plus = np.arange(1.0, nobs + 1) / nobs
            d_plus = (plus[:, None] - cdf).max(0)
            minus = np.arange(0.0, nobs) / nobs
            d_minus = (cdf - minus[:, None]).max(0)
            d = np.max(np.abs(np.c_[d_plus, d_minus]), 1)
            results[ss].append(d)
        logging.log(logging.INFO, 'Completed {}, remaining {}'.format(NUM_SIM - remaining, remaining))
        elapsed = dt.datetime.now() - start
        rem = elapsed.total_seconds() / (NUM_SIM - remaining) * remaining
        logging.log(logging.INFO, f'({sim_type}) Time remaining {rem:0.1f}s')
    for key in results:
        results[key] = np.concatenate(results[key])
    if save:
        file_name = f'lilliefors-sim-{sim_type}-results.pkl.gz'
        with gzip.open(file_name, 'wb', 5) as pkl:
            pickle.dump(results, pkl)
    crit_vals = {}
    for key in results:
        crit_vals[key] = np.percentile(results[key], PERCENTILES)
    start = 20
    num = len([k for k in crit_vals if k >= start])
    all_x = np.zeros((num * len(PERCENTILES), len(PERCENTILES) + 2))
    all_y = np.zeros(num * len(PERCENTILES))
    loc = 0
    for i, perc in enumerate(PERCENTILES):
        y = pd.DataFrame(results).quantile(perc / 100.0)
        y = y.loc[start:]
        all_y[loc:loc + len(y)] = np.log(y)
        x = y.index.values.astype(float)
        all_x[loc:loc + len(y), -2:] = np.c_[np.log(x), np.log(x) ** 2]
        all_x[loc:loc + len(y), i:i + 1] = 1
        loc += len(y)
    w = np.ones_like(all_y).reshape(len(PERCENTILES), -1)
    w[6:, -5:] = 3
    w = w.ravel()
    res = sm.WLS(all_y, all_x, weights=w).fit()
    params = []
    for i in range(len(PERCENTILES)):
        params.append(np.r_[res.params[i], res.params[-2:]])
    params = np.array(params)
    df = pd.DataFrame(params).T
    df.columns = PERCENTILES
    asymp_crit_vals = {}
    for col in df:
        asymp_crit_vals[col] = df[col].values
    code = f'{sim_type}_crit_vals = '
    code += str(crit_vals).strip() + '\n\n'
    code += '\n# Coefficients are model '
    code += 'log(cv) = b[0] + b[1] log(n) + b[2] log(n)**2\n'
    code += f'{sim_type}_asymp_crit_vals = '
    code += str(asymp_crit_vals) + '\n\n'
    return code