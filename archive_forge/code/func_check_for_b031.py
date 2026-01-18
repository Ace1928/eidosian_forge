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
def check_for_b031(self, loop_node):
    """Check that `itertools.groupby` isn't iterated over more than once.

        We emit a warning when the generator returned by `groupby()` is used
        more than once inside a loop body or when it's used in a nested loop.
        """
    if isinstance(loop_node.iter, ast.Call):
        node = loop_node.iter
        if isinstance(node.func, ast.Name) and node.func.id in ('groupby',) or (isinstance(node.func, ast.Attribute) and node.func.attr == 'groupby' and isinstance(node.func.value, ast.Name) and (node.func.value.id == 'itertools')):
            if isinstance(loop_node.target, ast.Tuple) and isinstance(loop_node.target.elts[1], ast.Name):
                group_name = loop_node.target.elts[1].id
            else:
                return
            num_usages = 0
            for node in walk_list(loop_node.body):
                if isinstance(node, ast.For):
                    for nested_node in walk_list(node.body):
                        assert nested_node != node
                        if isinstance(nested_node, ast.Name) and nested_node.id == group_name:
                            self.errors.append(B031(nested_node.lineno, nested_node.col_offset, vars=(nested_node.id,)))
                if isinstance(node, ast.Name) and node.id == group_name:
                    num_usages += 1
                    if num_usages > 1:
                        self.errors.append(B031(node.lineno, node.col_offset, vars=(node.id,)))