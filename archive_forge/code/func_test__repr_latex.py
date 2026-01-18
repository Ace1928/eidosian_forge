from numpy.testing import assert_equal
from statsmodels.iolib.table import SimpleTable, default_txt_fmt
from statsmodels.iolib.table import default_latex_fmt
from statsmodels.iolib.table import default_html_fmt
import pandas
from statsmodels.regression.linear_model import OLS
def test__repr_latex(self):
    desired = '\n\\begin{center}\n\\begin{tabular}{lcc}\n\\toprule\n               & \\textbf{header1} & \\textbf{header2}  \\\\\n\\midrule\n\\textbf{stub1} &      5.394       &       29.3        \\\\\n\\textbf{stub2} &       343        &       34.2        \\\\\n\\bottomrule\n\\end{tabular}\n\\end{center}\n'
    testdata = [[5.394, 29.3], [343, 34.2]]
    teststubs = ('stub1', 'stub2')
    testheader = ('header1', 'header2')
    tbl = SimpleTable(testdata, testheader, teststubs, txt_fmt=default_txt_fmt)
    actual = '\n%s\n' % tbl._repr_latex_()
    assert_equal(actual, desired)