import re
from inspect import signature
from typing import Optional
import pytest
from sklearn.experimental import (
from sklearn.utils.discovery import all_displays, all_estimators, all_functions
@pytest.mark.parametrize('function_name', get_all_functions_names())
def test_function_docstring(function_name, request):
    """Check function docstrings using numpydoc."""
    res = numpydoc_validation.validate(function_name)
    res['errors'] = list(filter_errors(res['errors'], method='function'))
    if res['errors']:
        msg = repr_errors(res, method=f'Tested function: {function_name}')
        raise ValueError(msg)