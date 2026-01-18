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
def check_for_b019(self, node):
    if len(node.decorator_list) == 0 or len(self.contexts) < 2 or (not isinstance(self.contexts[-2].node, ast.ClassDef)):
        return
    resolved_decorators = ('.'.join(compose_call_path(decorator)) for decorator in node.decorator_list)
    for idx, decorator in enumerate(resolved_decorators):
        if decorator in {'classmethod', 'staticmethod'}:
            return
        if decorator in B019.caches:
            self.errors.append(B019(node.decorator_list[idx].lineno, node.decorator_list[idx].col_offset))
            return