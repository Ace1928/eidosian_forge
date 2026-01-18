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
def check_for_b034(self, node: ast.Call):
    if not isinstance(node.func, ast.Attribute):
        return
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != 're':
        return

    def check(num_args, param_name):
        if len(node.args) > num_args:
            self.errors.append(B034(node.args[num_args].lineno, node.args[num_args].col_offset, vars=(node.func.attr, param_name)))
    if node.func.attr in ('sub', 'subn'):
        check(3, 'count')
    elif node.func.attr == 'split':
        check(2, 'maxsplit')