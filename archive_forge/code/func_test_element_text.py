import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
def test_element_text(self):
    et = ggplot2.element_text()
    assert isinstance(et, ggplot2.ElementText)