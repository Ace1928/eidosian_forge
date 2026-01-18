import numpy as np
from numpy.testing import assert_equal, assert_raises
from pandas import Series
import pytest
from statsmodels.graphics.factorplots import _recode, interaction_plot
@pytest.mark.matplotlib
def test_formatting_errors(self, close_figures):
    assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, markers=['D'])
    assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, colors=['b', 'r', 'g'])
    assert_raises(ValueError, interaction_plot, self.weight, self.duration, self.days, linestyles=['--', '-.', ':'])