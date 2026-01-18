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
def check_for_b025(self, node):
    seen = []
    for handler in node.handlers:
        if isinstance(handler.type, (ast.Name, ast.Attribute)):
            name = '.'.join(compose_call_path(handler.type))
            seen.append(name)
        elif isinstance(handler.type, ast.Tuple):
            uniques = set()
            for entry in handler.type.elts:
                name = '.'.join(compose_call_path(entry))
                uniques.add(name)
            seen.extend(uniques)
    duplicates = sorted({x for x in seen if seen.count(x) > 1})
    for duplicate in duplicates:
        self.errors.append(B025(node.lineno, node.col_offset, vars=(duplicate,)))