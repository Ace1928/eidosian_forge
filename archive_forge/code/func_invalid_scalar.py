import operator
import pytest
from pandas._config.config import _get_option
from pandas import (
@pytest.fixture
def invalid_scalar(data):
    """
    A scalar that *cannot* be held by this ExtensionArray.

    The default should work for most subclasses, but is not guaranteed.

    If the array can hold any item (i.e. object dtype), then use pytest.skip.
    """
    return object.__new__(object)