from __future__ import annotations
import logging
import re
import typing as t
from collections import defaultdict
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ErrorLevel, UnsupportedError, concat_messages
from sqlglot.helper import apply_index_offset, csv, seq_get
from sqlglot.jsonpath import ALL_JSON_PATH_PARTS, JSON_PATH_PART_TRANSFORMS
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def set_operations(self, expression: exp.Union) -> str:
    if not self.OUTER_UNION_MODIFIERS:
        limit = expression.args.get('limit')
        order = expression.args.get('order')
        if limit or order:
            select = exp.subquery(expression, '_l_0', copy=False).select('*', copy=False)
            if limit:
                select = select.limit(limit.pop(), copy=False)
            if order:
                select = select.order_by(order.pop(), copy=False)
            return self.sql(select)
    sqls: t.List[str] = []
    stack: t.List[t.Union[str, exp.Expression]] = [expression]
    while stack:
        node = stack.pop()
        if isinstance(node, exp.Union):
            stack.append(node.expression)
            stack.append(self.maybe_comment(getattr(self, f'{node.key}_op')(node), comments=node.comments, separated=True))
            stack.append(node.this)
        else:
            sqls.append(self.sql(node))
    this = self.sep().join(sqls)
    this = self.query_modifiers(expression, this)
    return self.prepend_ctes(expression, this)