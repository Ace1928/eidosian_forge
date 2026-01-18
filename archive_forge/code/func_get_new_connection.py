from functools import lru_cache
from django.db.backends.base.base import NO_DB_ALIAS
from django.db.backends.postgresql.base import DatabaseWrapper as PsycopgDatabaseWrapper
from django.db.backends.postgresql.features import (
from django.db.backends.postgresql.introspection import (
from django.db.backends.postgresql.operations import (
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from .adapter import PostGISAdapter
from .features import DatabaseFeatures
from .introspection import PostGISIntrospection
from .operations import PostGISOperations
from .schema import PostGISSchemaEditor
def get_new_connection(self, conn_params):
    connection = super().get_new_connection(conn_params)
    if is_psycopg3:
        self.register_geometry_adapters(connection)
    return connection