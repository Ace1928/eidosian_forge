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
def compose_call_path(node):
    if isinstance(node, ast.Attribute):
        yield from compose_call_path(node.value)
        yield node.attr
    elif isinstance(node, ast.Call):
        yield from compose_call_path(node.func)
    elif isinstance(node, ast.Name):
        yield node.id