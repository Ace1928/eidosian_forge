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
def check_for_b012(self, node):

    def _loop(node, bad_node_types):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            return
        if isinstance(node, (ast.While, ast.For)):
            bad_node_types = (ast.Return,)
        elif isinstance(node, bad_node_types):
            self.errors.append(B012(node.lineno, node.col_offset))
        for child in ast.iter_child_nodes(node):
            _loop(child, bad_node_types)
    for child in node.finalbody:
        _loop(child, (ast.Return, ast.Continue, ast.Break))