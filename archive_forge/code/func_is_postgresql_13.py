import operator
from django.db import DataError, InterfaceError
from django.db.backends.base.features import BaseDatabaseFeatures
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from django.utils.functional import cached_property
@cached_property
def is_postgresql_13(self):
    return self.connection.pg_version >= 130000