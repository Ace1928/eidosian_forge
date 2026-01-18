import functools
from django.db import connections
from django.db.backends.base.base import NO_DB_ALIAS
from django.db.backends.postgresql.psycopg_any import is_psycopg3
def register_type_handlers(connection, **kwargs):
    if connection.vendor != 'postgresql' or connection.alias == NO_DB_ALIAS:
        return
    oids, array_oids = get_hstore_oids(connection.alias)
    if oids:
        register_hstore(connection.connection, globally=True, oid=oids, array_oid=array_oids)
    oids, citext_oids = get_citext_oids(connection.alias)
    if oids:
        array_type = psycopg2.extensions.new_array_type(citext_oids, 'citext[]', psycopg2.STRING)
        psycopg2.extensions.register_type(array_type, None)