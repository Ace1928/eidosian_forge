import numpy as np
from scipy import stats
from scipy.special import comb
from scipy.stats.distributions import rv_continuous
import matplotlib.pyplot as plt
from numpy import where, inf
from numpy import abs as np_abs
def meanexcess(thresh, shape, scale):
    """mean excess function of genpareto

    assert are inequality conditions in de Zea Bermudez, Kotz
    """
    warnif(shape > -1, 'shape > -1')
    warnif(thresh >= 0, 'thresh >= 0')
    warnif(scale - shape * thresh > 0, '(scale - shape*thresh) > 0')
    return (scale - shape * thresh) / (1 + shape)