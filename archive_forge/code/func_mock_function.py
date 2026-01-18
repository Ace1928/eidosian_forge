import pickle
import pytest
from sklearn.utils.deprecation import _is_deprecated, deprecated
@deprecated()
def mock_function():
    return 10