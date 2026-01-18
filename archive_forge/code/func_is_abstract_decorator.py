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
def is_abstract_decorator(expr):
    return isinstance(expr, ast.Name) and expr.id[:8] == 'abstract' or (isinstance(expr, ast.Attribute) and expr.attr[:8] == 'abstract')