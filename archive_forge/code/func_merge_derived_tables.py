from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def merge_derived_tables(expression, leave_tables_isolated=False):
    for outer_scope in traverse_scope(expression):
        for subquery in outer_scope.derived_tables:
            from_or_join = subquery.find_ancestor(exp.From, exp.Join)
            alias = subquery.alias_or_name
            inner_scope = outer_scope.sources[alias]
            if _mergeable(outer_scope, inner_scope, leave_tables_isolated, from_or_join):
                _rename_inner_sources(outer_scope, inner_scope, alias)
                _merge_from(outer_scope, inner_scope, subquery, alias)
                _merge_expressions(outer_scope, inner_scope, alias)
                _merge_joins(outer_scope, inner_scope, from_or_join)
                _merge_where(outer_scope, inner_scope, from_or_join)
                _merge_order(outer_scope, inner_scope)
                _merge_hints(outer_scope, inner_scope)
                outer_scope.clear_cache()
    return expression