from peewee import *
from playhouse.cockroachdb import CockroachDatabase
from playhouse.cockroachdb import PooledCockroachDatabase
from playhouse.pool import PooledMySQLDatabase
from playhouse.pool import PooledPostgresqlDatabase
from playhouse.pool import PooledSqliteDatabase
from playhouse.pool import PooledSqliteExtDatabase
from playhouse.sqlite_ext import SqliteExtDatabase
def register_database(db_class, *names):
    global schemes
    for name in names:
        schemes[name] = db_class