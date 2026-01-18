from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def concat_exprs(node: t.Optional[exp.Expression], exprs: t.List[exp.Expression]) -> exp.Expression:
    if isinstance(node, exp.Distinct) and len(node.expressions) > 1:
        concat_exprs = [self.expression(exp.Concat, expressions=node.expressions, safe=True)]
        node.set('expressions', concat_exprs)
        return node
    if len(exprs) == 1:
        return exprs[0]
    return self.expression(exp.Concat, expressions=args, safe=True)