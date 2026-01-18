from statsmodels.compat.pandas import FUTURE_STACK
from statsmodels.compat.python import lzip
import datetime
from functools import reduce
import re
import textwrap
import numpy as np
import pandas as pd
from .table import SimpleTable
from .tableformatting import fmt_latex, fmt_txt
def summary_col(results, float_format='%.4f', model_names=(), stars=False, info_dict=None, regressor_order=(), drop_omitted=False, include_r2=True):
    """
    Summarize multiple results instances side-by-side (coefs and SEs)

    Parameters
    ----------
    results : statsmodels results instance or list of result instances
    float_format : str, optional
        float format for coefficients and standard errors
        Default : '%.4f'
    model_names : list[str], optional
        Must have same length as the number of results. If the names are not
        unique, a roman number will be appended to all model names
    stars : bool
        print significance stars
    info_dict : dict, default None
        dict of functions to be applied to results instances to retrieve
        model info. To use specific information for different models, add a
        (nested) info_dict with model name as the key.
        Example: `info_dict = {"N":lambda x:(x.nobs), "R2": ..., "OLS":{
        "R2":...}}` would only show `R2` for OLS regression models, but
        additionally `N` for all other results.
        Default : None (use the info_dict specified in
        result.default_model_infos, if this property exists)
    regressor_order : list[str], optional
        list of names of the regressors in the desired order. All regressors
        not specified will be appended to the end of the list.
    drop_omitted : bool, optional
        Includes regressors that are not specified in regressor_order. If
        False, regressors not specified will be appended to end of the list.
        If True, only regressors in regressor_order will be included.
    include_r2 : bool, optional
        Includes R2 and adjusted R2 in the summary table.
    """
    if not isinstance(results, list):
        results = [results]
    cols = [_col_params(x, stars=stars, float_format=float_format, include_r2=include_r2) for x in results]
    if model_names:
        colnames = _make_unique(model_names)
    else:
        colnames = _make_unique([x.columns[0] for x in cols])
    for i in range(len(cols)):
        cols[i].columns = [colnames[i]]

    def merg(x, y):
        return x.merge(y, how='outer', right_index=True, left_index=True)
    index = list(cols[0].index)
    for col in cols[1:]:
        for key in col.index:
            if key not in index:
                index.append(key)
    for special in (('R-squared', ''), ('R-squared Adj.', '')):
        if special in index:
            index.remove(special)
            index.insert(len(index), special)
    summ = reduce(merg, cols)
    summ = summ.reindex(index)
    if regressor_order:
        varnames = summ.index.get_level_values(0).tolist()
        vc = pd.Series(varnames).value_counts()
        varnames = vc.loc[vc == 2].index.tolist()
        ordered = [x for x in regressor_order if x in varnames]
        unordered = [x for x in varnames if x not in regressor_order]
        new_order = ordered + unordered
        other = [x for x in summ.index.get_level_values(0) if x not in new_order]
        new_order += other
        if drop_omitted:
            for uo in unordered:
                new_order.remove(uo)
        summ = summ.reindex(new_order, level=0)
    idx = []
    index = summ.index.get_level_values(0)
    for i in range(0, index.shape[0], 2):
        idx.append(index[i])
        if i + 1 < index.shape[0] and index[i] == index[i + 1]:
            idx.append('')
        else:
            idx.append(index[i + 1])
    summ.index = idx
    if info_dict:
        cols = [_col_info(x, info_dict.get(x.model.__class__.__name__, info_dict)) for x in results]
    else:
        cols = [_col_info(x, getattr(x, 'default_model_infos', None)) for x in results]
    for df, name in zip(cols, _make_unique([df.columns[0] for df in cols])):
        df.columns = [name]
    info = reduce(merg, cols)
    dat = pd.DataFrame(np.vstack([summ, info]))
    dat.columns = summ.columns
    dat.index = pd.Index(summ.index.tolist() + info.index.tolist())
    summ = dat
    summ = summ.fillna('')
    smry = Summary()
    smry._merge_latex = True
    smry.add_df(summ, header=True, align='l')
    smry.add_text('Standard errors in parentheses.')
    if stars:
        smry.add_text('* p<.1, ** p<.05, ***p<.01')
    return smry