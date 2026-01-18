import re
import sys
import numpy as np
import pytest
from matplotlib import _preprocess_data
from matplotlib.axes import Axes
from matplotlib.testing import subprocess_run_for_testing
from matplotlib.testing.decorators import check_figures_equal
def test_docstring_addition():

    @_preprocess_data()
    def funcy(ax, *args, **kwargs):
        """
        Parameters
        ----------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        """
    assert re.search('all parameters also accept a string', funcy.__doc__)
    assert not re.search('the following parameters', funcy.__doc__)

    @_preprocess_data(replace_names=[])
    def funcy(ax, x, y, z, bar=None):
        """
        Parameters
        ----------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        """
    assert not re.search('all parameters also accept a string', funcy.__doc__)
    assert not re.search('the following parameters', funcy.__doc__)

    @_preprocess_data(replace_names=['bar'])
    def funcy(ax, x, y, z, bar=None):
        """
        Parameters
        ----------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        """
    assert not re.search('all parameters also accept a string', funcy.__doc__)
    assert not re.search('the following parameters .*: \\*bar\\*\\.', funcy.__doc__)

    @_preprocess_data(replace_names=['x', 't'])
    def funcy(ax, x, y, z, t=None):
        """
        Parameters
        ----------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        """
    assert not re.search('all parameters also accept a string', funcy.__doc__)
    assert not re.search('the following parameters .*: \\*x\\*, \\*t\\*\\.', funcy.__doc__)