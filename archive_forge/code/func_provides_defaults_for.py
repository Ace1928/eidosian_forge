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
def provides_defaults_for(self, rule: Rule) -> bool:
    """Check if this rule has defaults for a given rule.

        :internal:
        """
    return bool(not self.build_only and self.defaults and (self.endpoint == rule.endpoint) and (self != rule) and (self.arguments == rule.arguments))