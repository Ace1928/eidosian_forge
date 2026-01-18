from numpy.testing import assert_equal
from statsmodels.iolib.table import SimpleTable, default_txt_fmt
from statsmodels.iolib.table import default_latex_fmt
from statsmodels.iolib.table import default_html_fmt
import pandas
from statsmodels.regression.linear_model import OLS
def test_ltx_fmt1(self):
    desired = '\n\\begin{tabular}{lcc}\n\\toprule\n               & \\textbf{header1} & \\textbf{header2}  \\\\\n\\midrule\n\\textbf{stub1} &       0.0        &        1          \\\\\n\\textbf{stub2} &        2         &      3.333        \\\\\n\\bottomrule\n\\end{tabular}\n'
    actual = '\n%s\n' % tbl.as_latex_tabular(center=False)
    assert_equal(actual, desired)
    desired_centered = '\n\\begin{center}\n%s\n\\end{center}\n' % desired[1:-1]
    actual_centered = '\n%s\n' % tbl.as_latex_tabular()
    assert_equal(actual_centered, desired_centered)