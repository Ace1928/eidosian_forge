from contextlib import contextmanager
from unittest.mock import patch
import pytest
from IPython.lib import latextools
from IPython.testing.decorators import (
from IPython.utils.process import FindCmdError
def test_latex_to_png_invalid_hex_colors():
    """
    Test that invalid hex colors provided to dvipng gives an exception.
    """
    latex_string = '$x^2$'
    pytest.raises(ValueError, lambda: latextools.latex_to_png(latex_string, backend='dvipng', color='#f00bar'))
    pytest.raises(ValueError, lambda: latextools.latex_to_png(latex_string, backend='dvipng', color='#f00'))