import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
def test_element_text_repr(self):
    et = ggplot2.element_text()
    assert repr(et).startswith('<instance of')