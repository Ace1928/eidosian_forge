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
def check_for_b905(self, node):
    if not (isinstance(node.func, ast.Name) and node.func.id == 'zip'):
        return
    for arg in node.args:
        if self._is_infinite_iterator(arg):
            return
    if not any((kw.arg == 'strict' for kw in node.keywords)):
        self.errors.append(B905(node.lineno, node.col_offset))