import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
from rpy2.robjects.vectors import StrVector
def test_mutate_at(self):
    dataf_a = dplyr.DataFrame(mtcars)
    dataf_b = dataf_a.mutate_at(StrVector(['gear']), rl('sqrt'))
    assert type(dataf_b) is dplyr.DataFrame