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
def check_for_b006_and_b008(self, node):
    visitor = FuntionDefDefaultsVisitor(self.b008_extend_immutable_calls)
    visitor.visit(node.args.defaults + node.args.kw_defaults)
    self.errors.extend(visitor.errors)