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
def in_class_init(self) -> bool:
    return len(self.contexts) >= 2 and isinstance(self.contexts[-2].node, ast.ClassDef) and isinstance(self.contexts[-1].node, ast.FunctionDef) and (self.contexts[-1].node.name == '__init__')