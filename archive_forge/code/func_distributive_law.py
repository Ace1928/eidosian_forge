from __future__ import annotations
import logging
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import while_changing
from sqlglot.optimizer.scope import find_all_in_scope
from sqlglot.optimizer.simplify import flatten, rewrite_between, uniq_sort
def distributive_law(expression, dnf, max_distance):
    """
    x OR (y AND z) -> (x OR y) AND (x OR z)
    (x AND y) OR (y AND z) -> (x OR y) AND (x OR z) AND (y OR y) AND (y OR z)
    """
    if normalized(expression, dnf=dnf):
        return expression
    distance = normalization_distance(expression, dnf=dnf)
    if distance > max_distance:
        raise OptimizeError(f'Normalization distance {distance} exceeds max {max_distance}')
    exp.replace_children(expression, lambda e: distributive_law(e, dnf, max_distance))
    to_exp, from_exp = (exp.Or, exp.And) if dnf else (exp.And, exp.Or)
    if isinstance(expression, from_exp):
        a, b = expression.unnest_operands()
        from_func = exp.and_ if from_exp == exp.And else exp.or_
        to_func = exp.and_ if to_exp == exp.And else exp.or_
        if isinstance(a, to_exp) and isinstance(b, to_exp):
            if len(tuple(a.find_all(exp.Connector))) > len(tuple(b.find_all(exp.Connector))):
                return _distribute(a, b, from_func, to_func)
            return _distribute(b, a, from_func, to_func)
        if isinstance(a, to_exp):
            return _distribute(b, a, from_func, to_func)
        if isinstance(b, to_exp):
            return _distribute(a, b, from_func, to_func)
    return expression