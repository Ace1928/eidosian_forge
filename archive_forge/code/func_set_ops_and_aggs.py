from __future__ import annotations
import math
import typing as t
from sqlglot import alias, exp
from sqlglot.helper import name_sequence
from sqlglot.optimizer.eliminate_joins import join_condition
def set_ops_and_aggs(step):
    step.operands = tuple((alias(operand, alias_) for operand, alias_ in operands.items()))
    step.aggregations = list(aggregations)