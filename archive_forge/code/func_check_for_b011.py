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
def check_for_b011(self, node):
    if isinstance(node.test, ast.Constant) and node.test.value is False:
        self.errors.append(B011(node.lineno, node.col_offset))