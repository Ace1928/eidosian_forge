import numpy as np
from scipy import stats
from statsmodels.sandbox.distributions.extras import (SkewNorm_gen, skewnorm,
from statsmodels.stats.moment_helpers import mc2mvsk, mnc2mc, mvsk2mnc
def examples_normexpand():
    skewnorm = SkewNorm_gen()
    rvs = skewnorm.rvs(5, size=100)
    normexpan = NormExpan_gen(rvs, mode='sample')
    smvsk = stats.describe(rvs)[2:]
    print('sample: mu,sig,sk,kur')
    print(smvsk)
    dmvsk = normexpan.stats(moments='mvsk')
    print('normexpan: mu,sig,sk,kur')
    print(dmvsk)
    print('mvsk diff distribution - sample')
    print(np.array(dmvsk) - np.array(smvsk))
    print('normexpan attributes mvsk')
    print(mc2mvsk(normexpan.cnt))
    print(normexpan.mvsk)
    mnc = mvsk2mnc(dmvsk)
    mc = mnc2mc(mnc)
    print('central moments')
    print(mc)
    print('non-central moments')
    print(mnc)
    pdffn = pdf_moments(mc)
    print('\npdf approximation from moments')
    print('pdf at', mc[0] - 1, mc[0] + 1)
    print(pdffn([mc[0] - 1, mc[0] + 1]))
    print(normexpan.pdf([mc[0] - 1, mc[0] + 1]))