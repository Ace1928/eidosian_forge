import functools
from django.db import connections
from django.db.backends.base.base import NO_DB_ALIAS
from django.db.backends.postgresql.psycopg_any import is_psycopg3
def get_type_oids(connection_alias, type_name):
    with connections[connection_alias].cursor() as cursor:
        cursor.execute('SELECT oid, typarray FROM pg_type WHERE typname = %s', (type_name,))
        oids = []
        array_oids = []
        for row in cursor:
            oids.append(row[0])
            array_oids.append(row[1])
        return (tuple(oids), tuple(array_oids))