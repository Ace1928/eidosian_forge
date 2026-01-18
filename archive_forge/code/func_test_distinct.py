import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
from rpy2.robjects.vectors import StrVector
def test_distinct(self):
    dataf_a = dplyr.DataFrame(mtcars)
    res = dataf_a.distinct()
    assert res.nrow == dataf_a.nrow