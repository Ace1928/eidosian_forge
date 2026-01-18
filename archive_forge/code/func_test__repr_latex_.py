import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
def test__repr_latex_(self):
    desired = '\n\\begin{table}\n\\caption{}\n\\label{}\n\\begin{center}\n\\begin{tabular}{lll}\n\\hline\n               & y I      & y II      \\\\\n\\hline\nconst          & 7.7500   & 12.4231   \\\\\n               & (1.1058) & (3.1872)  \\\\\nx1             & -0.7500  & -1.5769   \\\\\n               & (0.2368) & (0.6826)  \\\\\nR-squared      & 0.7697   & 0.6401    \\\\\nR-squared Adj. & 0.6930   & 0.5202    \\\\\n\\hline\n\\end{tabular}\n\\end{center}\n\\end{table}\n\\bigskip\nStandard errors in parentheses.\n'
    x = [1, 5, 7, 3, 5]
    x = add_constant(x)
    y1 = [6, 4, 2, 7, 4]
    y2 = [8, 5, 0, 12, 4]
    reg1 = OLS(y1, x).fit()
    reg2 = OLS(y2, x).fit()
    actual = summary_col([reg1, reg2])._repr_latex_()
    actual = '\n%s\n' % actual
    assert_equal(actual, desired)