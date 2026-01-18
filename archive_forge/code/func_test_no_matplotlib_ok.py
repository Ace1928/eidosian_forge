import sys
import types
import pytest
import pandas.util._test_decorators as td
import pandas
@td.skip_if_installed('matplotlib')
def test_no_matplotlib_ok():
    msg = 'matplotlib is required for plotting when the default backend "matplotlib" is selected.'
    with pytest.raises(ImportError, match=msg):
        pandas.plotting._core._get_plot_backend('matplotlib')