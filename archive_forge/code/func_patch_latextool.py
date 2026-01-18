from contextlib import contextmanager
from unittest.mock import patch
import pytest
from IPython.lib import latextools
from IPython.testing.decorators import (
from IPython.utils.process import FindCmdError
@contextmanager
def patch_latextool(mock=mock_kpsewhich):
    with patch.object(latextools, 'kpsewhich', mock):
        yield