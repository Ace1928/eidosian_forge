from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
@pytest.mark.parametrize('data,res_expect_str,atol', ((data_same_size, sas_same_size, 0.0001), (data_diff_size, sas_diff_size, 0.0001), (extreme_size, sas_extreme, 1e-10)), ids=['equal size sample', 'unequal sample size', 'extreme sample size differences'])
def test_compare_sas(self, data, res_expect_str, atol):
    """
        SAS code used to generate results for each sample:
        DATA ACHE;
        INPUT BRAND RELIEF;
        CARDS;
        1 24.5
        ...
        3 27.8
        ;
        ods graphics on;   ODS RTF;ODS LISTING CLOSE;
           PROC ANOVA DATA=ACHE;
           CLASS BRAND;
           MODEL RELIEF=BRAND;
           MEANS BRAND/TUKEY CLDIFF;
           TITLE 'COMPARE RELIEF ACROSS MEDICINES  - ANOVA EXAMPLE';
           ods output  CLDiffs =tc;
        proc print data=tc;
            format LowerCL 17.16 UpperCL 17.16 Difference 17.16;
            title "Output with many digits";
        RUN;
        QUIT;
        ODS RTF close;
        ODS LISTING;
        """
    res_expect = np.asarray(res_expect_str.replace(' - ', ' ').split()[5:], dtype=float).reshape((6, 6))
    res_tukey = stats.tukey_hsd(*data)
    conf = res_tukey.confidence_interval()
    for i, j, l, s, h, sig in res_expect:
        i, j = (int(i) - 1, int(j) - 1)
        assert_allclose(conf.low[i, j], l, atol=atol)
        assert_allclose(res_tukey.statistic[i, j], s, atol=atol)
        assert_allclose(conf.high[i, j], h, atol=atol)
        assert_allclose(res_tukey.pvalue[i, j] <= 0.05, sig == 1)