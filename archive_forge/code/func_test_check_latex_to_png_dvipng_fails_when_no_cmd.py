from contextlib import contextmanager
from unittest.mock import patch
import pytest
from IPython.lib import latextools
from IPython.testing.decorators import (
from IPython.utils.process import FindCmdError
@pytest.mark.parametrize('command', ['latex', 'dvipng'])
def test_check_latex_to_png_dvipng_fails_when_no_cmd(command):

    def mock_find_cmd(arg):
        if arg == command:
            raise FindCmdError
    with patch.object(latextools, 'find_cmd', mock_find_cmd):
        assert latextools.latex_to_png_dvipng('whatever', True) is None