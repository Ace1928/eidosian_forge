from __future__ import annotations
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from sqlglot import Dialect, expressions as exp
from sqlglot.helper import ensure_list
def compute_node_mappings(original: exp.Expression, copy: exp.Expression) -> t.Dict[int, exp.Expression]:
    return {id(old_node): new_node for old_node, new_node in zip(original.walk(), copy.walk()) if id(old_node) in matching_ids}