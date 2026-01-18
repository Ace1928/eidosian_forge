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
def check_for_b007(self, node):
    targets = NameFinder()
    targets.visit(node.target)
    ctrl_names = set(filter(lambda s: not s.startswith('_'), targets.names))
    body = NameFinder()
    for expr in node.body:
        body.visit(expr)
    used_names = set(body.names)
    for name in sorted(ctrl_names - used_names):
        n = targets.names[name][0]
        self.errors.append(B007(n.lineno, n.col_offset, vars=(name,)))