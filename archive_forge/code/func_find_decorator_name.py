from __future__ import annotations
import ast
import builtins
import itertools
import logging
import math
import re
import sys
import warnings
from collections import namedtuple
from contextlib import suppress
from functools import lru_cache, partial
from keyword import iskeyword
from typing import Dict, List, Set, Union
import attr
import pycodestyle
@classmethod
def find_decorator_name(cls, d):
    if isinstance(d, ast.Name):
        return d.id
    elif isinstance(d, ast.Attribute):
        return d.attr
    elif isinstance(d, ast.Call):
        return cls.find_decorator_name(d.func)