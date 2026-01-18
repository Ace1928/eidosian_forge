import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('Encoder', [OneHotEncoder, OrdinalEncoder])
def test_encoders_has_categorical_tags(Encoder):
    assert 'categorical' in Encoder()._get_tags()['X_types']