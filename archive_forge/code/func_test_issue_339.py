from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
def test_issue_339():
    data = load_anes96()
    exog = data.exog
    exog = exog[:, :-1]
    exog = sm.add_constant(exog, prepend=True)
    res1 = sm.MNLogit(data.endog, exog).fit(method='newton', disp=0)
    smry = '\n'.join(res1.summary().as_text().split('\n')[9:])
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_case_file = os.path.join(cur_dir, 'results', 'mn_logit_summary.txt')
    with open(test_case_file, encoding='utf-8') as fd:
        test_case = fd.read()
    np.testing.assert_equal(smry, test_case[:-1])
    res1.summary2()