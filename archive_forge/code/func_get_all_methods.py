import re
from inspect import signature
from typing import Optional
import pytest
from sklearn.experimental import (
from sklearn.utils.discovery import all_displays, all_estimators, all_functions
def get_all_methods():
    estimators = all_estimators()
    displays = all_displays()
    for name, Klass in estimators + displays:
        if name.startswith('_'):
            continue
        methods = []
        for name in dir(Klass):
            if name.startswith('_'):
                continue
            method_obj = getattr(Klass, name)
            if hasattr(method_obj, '__call__') or isinstance(method_obj, property):
                methods.append(name)
        methods.append(None)
        for method in sorted(methods, key=str):
            yield (Klass, method)