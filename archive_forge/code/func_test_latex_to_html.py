from contextlib import contextmanager
from unittest.mock import patch
import pytest
from IPython.lib import latextools
from IPython.testing.decorators import (
from IPython.utils.process import FindCmdError
@skipif_not_matplotlib
def test_latex_to_html():
    img = latextools.latex_to_html('$x^2$')
    assert 'data:image/png;base64,iVBOR' in img