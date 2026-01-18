from contextlib import contextmanager
from unittest.mock import patch
import pytest
from IPython.lib import latextools
from IPython.testing.decorators import (
from IPython.utils.process import FindCmdError
@pytest.mark.parametrize('context', [no_op, patch_latextool])
@pytest.mark.parametrize('s_wrap', [('$x^2$', False), ('x^2', True)])
def test_latex_to_png_mpl_runs(s_wrap, context):
    """
    Test that latex_to_png_mpl just runs without error.
    """
    try:
        import matplotlib
    except ImportError:
        pytest.skip('This needs matplotlib to be available')
        return
    s, wrap = s_wrap
    with context():
        latextools.latex_to_png_mpl(s, wrap)