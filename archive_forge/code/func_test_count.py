import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
from rpy2.robjects.vectors import StrVector
def test_count(self):
    dataf_a = dplyr.DataFrame(mtcars)
    res = dataf_a.count()
    assert tuple(res.rx2('n')) == (dataf_a.nrow,)