from collections import defaultdict
from sqlglot import alias, exp
from sqlglot.optimizer.qualify_columns import Resolver
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import ensure_schema
def default_selection(is_agg: bool) -> exp.Alias:
    return alias(exp.Max(this=exp.Literal.number(1)) if is_agg else '1', '_')