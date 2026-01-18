from __future__ import annotations
import ast
import re
import typing as t
from dataclasses import dataclass
from string import Template
from types import CodeType
from urllib.parse import quote
from ..datastructures import iter_multi_items
from ..urls import _urlencode
from .converters import ValidationError
def suitable_for(self, values: t.Mapping[str, t.Any], method: str | None=None) -> bool:
    """Check if the dict of values has enough data for url generation.

        :internal:
        """
    if method is not None and self.methods is not None and (method not in self.methods):
        return False
    defaults = self.defaults or ()
    for key in self.arguments:
        if key not in defaults and key not in values:
            return False
    if defaults:
        for key, value in defaults.items():
            if key in values and value != values[key]:
                return False
    return True