import numpy as np
import numpy.testing as npt
import pytest
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.extra.array_api import make_strategies_namespace
from xarray.core.variable import Variable
from xarray.testing.strategies import (
from xarray.tests import requires_numpy_array_api
def check_dict_values(dictionary: dict, allowed_attrs_values_types) -> bool:
    """Helper function to assert that all values in recursive dict match one of a set of types."""
    for key, value in dictionary.items():
        if isinstance(value, allowed_attrs_values_types) or value is None:
            continue
        elif isinstance(value, dict):
            if not check_dict_values(value, allowed_attrs_values_types):
                return False
        else:
            return False
    return True