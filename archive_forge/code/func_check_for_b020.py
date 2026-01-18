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
def check_for_b020(self, node):
    targets = NameFinder()
    targets.visit(node.target)
    ctrl_names = set(targets.names)
    iterset = B020NameFinder()
    iterset.visit(node.iter)
    iterset_names = set(iterset.names)
    for name in sorted(ctrl_names):
        if name in iterset_names:
            n = targets.names[name][0]
            self.errors.append(B020(n.lineno, n.col_offset, vars=(name,)))