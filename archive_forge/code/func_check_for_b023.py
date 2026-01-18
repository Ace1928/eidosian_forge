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
def check_for_b023(self, loop_node):
    """Check that functions (including lambdas) do not use loop variables.

        https://docs.python-guide.org/writing/gotchas/#late-binding-closures from
        functions - usually but not always lambdas - defined inside a loop are a
        classic source of bugs.

        For each use of a variable inside a function defined inside a loop, we
        emit a warning if that variable is reassigned on each loop iteration
        (outside the function).  This includes but is not limited to explicit
        loop variables like the `x` in `for x in range(3):`.
        """
    safe_functions = []
    suspicious_variables = []
    for node in ast.walk(loop_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ('filter', 'reduce', 'map') or (isinstance(node.func, ast.Attribute) and node.func.attr == 'reduce' and isinstance(node.func.value, ast.Name) and (node.func.value.id == 'functools')):
                for arg in node.args:
                    if isinstance(arg, FUNCTION_NODES):
                        safe_functions.append(arg)
            for keyword in node.keywords:
                if keyword.arg == 'key' and isinstance(keyword.value, FUNCTION_NODES):
                    safe_functions.append(keyword.value)
        if isinstance(node, ast.Return):
            if isinstance(node.value, FUNCTION_NODES):
                safe_functions.append(node.value)
        if isinstance(node, FUNCTION_NODES) and node not in safe_functions:
            argnames = {arg.arg for arg in ast.walk(node.args) if isinstance(arg, ast.arg)}
            if isinstance(node, ast.Lambda):
                body_nodes = ast.walk(node.body)
            else:
                body_nodes = itertools.chain.from_iterable(map(ast.walk, node.body))
            errors = []
            for name in body_nodes:
                if isinstance(name, ast.Name) and name.id not in argnames:
                    if isinstance(name.ctx, ast.Load):
                        errors.append(B023(name.lineno, name.col_offset, vars=(name.id,)))
                    elif isinstance(name.ctx, ast.Store):
                        argnames.add(name.id)
            for err in errors:
                if err.vars[0] not in argnames and err not in self._b023_seen:
                    self._b023_seen.add(err)
                    suspicious_variables.append(err)
    if suspicious_variables:
        reassigned_in_loop = set(self._get_assigned_names(loop_node))
    for err in sorted(suspicious_variables):
        if reassigned_in_loop.issuperset(err.vars):
            self.errors.append(err)