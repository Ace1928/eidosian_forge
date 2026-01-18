from numpy.testing import assert_equal
from statsmodels.iolib.table import SimpleTable, default_txt_fmt
from statsmodels.iolib.table import default_latex_fmt
from statsmodels.iolib.table import default_html_fmt
import pandas
from statsmodels.regression.linear_model import OLS
def test_html_fmt1(self):
    desired = '\n<table class="simpletable">\n<tr>\n    <td></td>    <th>header1</th> <th>header2</th>\n</tr>\n<tr>\n  <th>stub1</th>   <td>0.0</td>      <td>1</td>   \n</tr>\n<tr>\n  <th>stub2</th>    <td>2</td>     <td>3.333</td> \n</tr>\n</table>\n'
    actual = '\n%s\n' % tbl.as_html()
    assert_equal(actual, desired)