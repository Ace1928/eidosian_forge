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
def check_for_b032(self, node):
    if node.value is None and hasattr(node.target, 'value') and isinstance(node.target.value, ast.Name) and (isinstance(node.target, ast.Subscript) or (isinstance(node.target, ast.Attribute) and node.target.value.id != 'self')):
        self.errors.append(B032(node.lineno, node.col_offset))