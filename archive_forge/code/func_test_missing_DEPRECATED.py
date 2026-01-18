import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
def test_missing_DEPRECATED(self):
    decorate = deprecate_parameter('old', start_version='0.10', stop_version='0.12', stacklevel=2)

    def foo(arg0, old=None):
        return (arg0, old)
    with pytest.raises(RuntimeError, match='Expected .* <DEPRECATED>'):
        decorate(foo)

    def bar(arg0, old=DEPRECATED):
        return arg0
    assert decorate(bar)(1) == 1