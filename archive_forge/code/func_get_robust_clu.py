import os
import numpy as np
import pandas as pd
import pytest
import statsmodels.discrete.discrete_model as smd
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.regression.linear_model import OLS
from statsmodels.base.covtype import get_robustcov_results
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import add_constant
from numpy.testing import assert_allclose, assert_equal, assert_
import statsmodels.tools._testing as smt
from .results import results_count_robust_cluster as results_st
@classmethod
def get_robust_clu(cls):
    res1 = cls.res1
    cov_clu = sw.cov_cluster(res1, group)
    cls.bse_rob = sw.se_cov(cov_clu)
    cls.corr_fact = cls.get_correction_factor(res1)