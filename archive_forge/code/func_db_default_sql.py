import logging
import operator
from datetime import datetime
from django.conf import settings
from django.db.backends.ddl_references import (
from django.db.backends.utils import names_digest, split_identifier, truncate_name
from django.db.models import NOT_PROVIDED, Deferrable, Index
from django.db.models.sql import Query
from django.db.transaction import TransactionManagementError, atomic
from django.utils import timezone
def db_default_sql(self, field):
    """Return the sql and params for the field's database default."""
    from django.db.models.expressions import Value
    db_default = field._db_default_expression
    sql = self._column_default_sql(field) if isinstance(db_default, Value) else '(%s)'
    query = Query(model=field.model)
    compiler = query.get_compiler(connection=self.connection)
    default_sql, params = compiler.compile(db_default)
    if self.connection.features.requires_literal_defaults:
        default_sql %= tuple((self.prepare_default(p) for p in params))
        params = []
    return (sql % default_sql, params)