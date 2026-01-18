import json
import os
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
def test_het_goldfeldquandt(self):
    het_gq_greater = dict(statistic=0.5313259064778423, pvalue=0.9990217851193723, parameters=(98, 98), distr='f')
    het_gq_less = dict(statistic=0.5313259064778423, pvalue=0.000978214880627621, parameters=(98, 98), distr='f')
    het_gq_two_sided = dict(statistic=0.5313259064778423, pvalue=0.001956429761255241, parameters=(98, 98), distr='f')
    het_gq_two_sided_01 = dict(statistic=0.5006976835928314, pvalue=0.001387126702579789, parameters=(88, 87), distr='f')
    endogg, exogg = (self.endog, self.exog)
    gq = smsdia.het_goldfeldquandt(endogg, exogg, split=0.5)
    compare_to_reference(gq, het_gq_greater, decimal=(12, 12))
    assert_equal(gq[-1], 'increasing')
    gq = smsdia.het_goldfeldquandt(endogg, exogg, split=0.5, alternative='decreasing')
    compare_to_reference(gq, het_gq_less, decimal=(12, 12))
    assert_equal(gq[-1], 'decreasing')
    gq = smsdia.het_goldfeldquandt(endogg, exogg, split=0.5, alternative='two-sided')
    compare_to_reference(gq, het_gq_two_sided, decimal=(12, 12))
    assert_equal(gq[-1], 'two-sided')
    gq = smsdia.het_goldfeldquandt(endogg, exogg, split=90, drop=21, alternative='two-sided')
    compare_to_reference(gq, het_gq_two_sided_01, decimal=(12, 12))
    assert_equal(gq[-1], 'two-sided')