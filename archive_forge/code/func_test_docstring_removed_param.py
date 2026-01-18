import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
@pytest.mark.skipif(not have_numpydoc, reason='requires numpydoc')
def test_docstring_removed_param(self):
    assert _func_deprecated_params.__name__ == '_func_deprecated_params'
    if sys.flags.optimize < 2:
        assert _func_deprecated_params.__doc__ == 'Expected docstring.\n\n\n    Parameters\n    ----------\n    arg0 : int\n        First unchanged parameter.\n    arg1 : int, optional\n        Second unchanged parameter.\n\n    Other Parameters\n    ----------------\n    old0 : DEPRECATED\n        `old0` is deprecated.\n\n        .. deprecated:: 0.10\n    old1 : DEPRECATED\n        `old1` is deprecated.\n\n        .. deprecated:: 0.10\n'