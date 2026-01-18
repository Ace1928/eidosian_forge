from django.core import checks
from django.db import connections, router
from django.db.models.sql import Query
from django.utils.functional import cached_property
from . import NOT_PROVIDED, Field
def generated_sql(self, connection):
    compiler = connection.ops.compiler('SQLCompiler')(self._query, connection=connection, using=None)
    resolved_expression = self.expression.resolve_expression(self._query, allow_joins=False)
    sql, params = compiler.compile(resolved_expression)
    if getattr(self.expression, 'conditional', False) and (not connection.features.supports_boolean_expr_in_select_clause):
        sql = f'CASE WHEN {sql} THEN 1 ELSE 0 END'
    return (sql, params)