from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
@custom_format.setter
def custom_format(self, val):
    if val is None:
        self._custom_format = {}
    elif isinstance(val, dict):
        for k, v in val.items():
            self._validate_function(f'custom_value.{k}', v)
        self._custom_format = val
    elif hasattr(val, '__call__'):
        self._validate_function('custom_value', val)
        for field in self._field_names:
            self._custom_format[field] = val
    else:
        msg = 'The custom_format property need to be a dictionary or callable'
        raise TypeError(msg)