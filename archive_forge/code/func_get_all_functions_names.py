import re
from inspect import signature
from typing import Optional
import pytest
from sklearn.experimental import (
from sklearn.utils.discovery import all_displays, all_estimators, all_functions
def get_all_functions_names():
    functions = all_functions()
    for _, func in functions:
        if 'utils.fixes' not in func.__module__:
            yield f'{func.__module__}.{func.__name__}'