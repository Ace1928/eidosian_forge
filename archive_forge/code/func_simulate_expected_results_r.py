from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
@pytest.fixture
def simulate_expected_results_r():
    """
    obtained from ets.simulate in the R package forecast, data is from fpp2
    package.

    library(magrittr)
    library(fpp2)
    library(forecast)
    concat <- function(...) {
      return(paste(..., sep=""))
    }
    error <- c("A", "M")
    trend <- c("A", "M", "N")
    seasonal <- c("A", "M", "N")
    models <- outer(error, trend, FUN = "concat") %>%
      outer(seasonal, FUN = "concat") %>% as.vector
    # innov from np.random.seed(0); np.random.randn(4)
    innov <- c(1.76405235, 0.40015721, 0.97873798, 2.2408932)
    params <- expand.grid(models, c(TRUE, FALSE))
    results <- apply(params, 1, FUN = function(p) {
      tryCatch(
        simulate(ets(austourists, model = p[1], damped = as.logical(p[2])),
                 innov = innov),
        error = function(e) c(NA, NA, NA, NA))
    }) %>% t
    rownames(results) <- apply(params, 1, FUN = function(x) paste(x[1], x[2]))
    """
    damped = {'AAA': [77.84173, 52.69818, 65.83254, 71.85204], 'MAA': [207.81653, 136.977, 253.56234, 588.958], 'MAM': [215.83822, 127.17132, 269.09483, 704.32105], 'MMM': [216.52591, 132.47637, 283.04889, 759.08043], 'AAN': [62.51423, 61.87381, 63.14735, 65.1136], 'MAN': [168.25189, 90.46201, 133.54769, 232.81738], 'MMN': [167.97747, 90.59675, 134.203, 235.64502]}
    undamped = {'AAA': [77.1086, 51.51669, 64.46857, 70.36349], 'MAA': [209.23158, 149.62943, 270.65579, 637.03828], 'ANA': [77.0932, 51.52384, 64.36231, 69.84786], 'MNA': [207.86986, 169.42706, 313.9796, 793.97948], 'MAM': [214.4575, 106.19605, 211.61304, 492.12223], 'MMM': [221.01861, 158.55914, 403.22625, 1389.33384], 'MNM': [215.00997, 140.93035, 309.92465, 875.07985], 'AAN': [63.66619, 63.09571, 64.45832, 66.51967], 'MAN': [172.37584, 91.51932, 134.11221, 230.9897], 'MMN': [169.88595, 97.33527, 142.97017, 252.51834], 'ANN': [60.53589, 59.51851, 60.1757, 61.63011], 'MNN': [163.01575, 112.58317, 172.21992, 338.93918]}
    return {True: damped, False: undamped}