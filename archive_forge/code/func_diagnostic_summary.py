from __future__ import annotations
from statsmodels.compat.pandas import (
from collections.abc import Iterable
import datetime
import datetime as dt
from types import SimpleNamespace
from typing import Any, Literal, cast
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import eval_measures
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
from statsmodels.tools.validation import (
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import (
from statsmodels.tsa.tsatools import freq_to_period, lagmat
import warnings
def diagnostic_summary(self):
    """
        Returns a summary containing standard model diagnostic tests

        Returns
        -------
        Summary
            A summary instance with panels for serial correlation tests,
            normality tests and heteroskedasticity tests.

        See Also
        --------
        test_serial_correlation
            Test models residuals for serial correlation.
        test_normality
            Test models residuals for deviations from normality.
        test_heteroskedasticity
            Test models residuals for conditional heteroskedasticity.
        """
    from statsmodels.iolib.table import SimpleTable
    spacer = SimpleTable([''])
    smry = Summary()
    sc = self.test_serial_correlation()
    sc = sc.loc[sc.DF > 0]
    values = [[i + 1] + row for i, row in enumerate(sc.values.tolist())]
    data_fmts = ('%10d', '%10.3f', '%10.3f', '%10d')
    if sc.shape[0]:
        tab = SimpleTable(values, headers=['Lag'] + list(sc.columns), title='Test of No Serial Correlation', header_align='r', data_fmts=data_fmts)
        smry.tables.append(tab)
        smry.tables.append(spacer)
    jb = self.test_normality()
    data_fmts = ('%10.3f', '%10.3f', '%10.3f', '%10.3f')
    tab = SimpleTable([jb.values], headers=list(jb.index), title='Test of Normality', header_align='r', data_fmts=data_fmts)
    smry.tables.append(tab)
    smry.tables.append(spacer)
    arch_lm = self.test_heteroskedasticity()
    values = [[i + 1] + row for i, row in enumerate(arch_lm.values.tolist())]
    data_fmts = ('%10d', '%10.3f', '%10.3f', '%10d')
    tab = SimpleTable(values, headers=['Lag'] + list(arch_lm.columns), title='Test of Conditional Homoskedasticity', header_align='r', data_fmts=data_fmts)
    smry.tables.append(tab)
    return smry