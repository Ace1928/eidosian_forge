import datetime
import os
import sys
from getpass import getpass
from optparse import OptionParser
from peewee import *
from peewee import print_
from peewee import __version__ as peewee_version
from playhouse.cockroachdb import CockroachDatabase
from playhouse.reflection import *
def make_introspector(database_type, database_name, **kwargs):
    if database_type not in DATABASE_MAP:
        err('Unrecognized database, must be one of: %s' % ', '.join(DATABASE_MAP.keys()))
        sys.exit(1)
    schema = kwargs.pop('schema', None)
    DatabaseClass = DATABASE_MAP[database_type]
    db = DatabaseClass(database_name, **kwargs)
    return Introspector.from_database(db, schema=schema)