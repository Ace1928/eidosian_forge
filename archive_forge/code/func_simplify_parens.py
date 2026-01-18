from __future__ import annotations
import datetime
import functools
import itertools
import typing as t
from collections import deque
from decimal import Decimal
from functools import reduce
import sqlglot
from sqlglot import Dialect, exp
from sqlglot.helper import first, merge_ranges, while_changing
from sqlglot.optimizer.scope import find_all_in_scope, walk_in_scope
def simplify_parens(expression):
    if not isinstance(expression, exp.Paren):
        return expression
    this = expression.this
    parent = expression.parent
    parent_is_predicate = isinstance(parent, exp.Predicate)
    if not isinstance(this, exp.Select) and (not isinstance(parent, exp.SubqueryPredicate)) and (not isinstance(parent, (exp.Condition, exp.Binary)) or isinstance(parent, exp.Paren) or (not isinstance(this, exp.Binary) and (not (isinstance(this, (exp.Not, exp.Is)) and parent_is_predicate))) or (isinstance(this, exp.Predicate) and (not parent_is_predicate)) or (isinstance(this, exp.Add) and isinstance(parent, exp.Add)) or (isinstance(this, exp.Mul) and isinstance(parent, exp.Mul)) or (isinstance(this, exp.Mul) and isinstance(parent, (exp.Add, exp.Sub)))):
        return this
    return expression