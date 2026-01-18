import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
def test_conflicting_old_and_new(self):
    match = '.*`old[0,1]` is deprecated'
    with pytest.warns(FutureWarning, match=match):
        with pytest.raises(ValueError, match='.* avoid conflicting values'):
            _func_replace_params(1, old0=2, new1=2)
    with pytest.warns(FutureWarning, match=match):
        with pytest.raises(ValueError, match='.* avoid conflicting values'):
            _func_replace_params(1, old1=2, new0=2)
    with pytest.warns(FutureWarning, match=match):
        with pytest.raises(ValueError, match='.* avoid conflicting values'):
            _func_replace_params(1, old0=1, old1=1, new0=1, new1=1)