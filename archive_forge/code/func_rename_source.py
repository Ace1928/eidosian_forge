from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
def rename_source(self, old_name, new_name):
    """Rename a source in this scope"""
    columns = self.sources.pop(old_name or '', [])
    self.sources[new_name] = columns