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
def get_option_parser():
    parser = OptionParser(usage='usage: %prog [options] database_name')
    ao = parser.add_option
    ao('-H', '--host', dest='host')
    ao('-p', '--port', dest='port', type='int')
    ao('-u', '--user', dest='user')
    ao('-P', '--password', dest='password', action='store_true')
    engines = sorted(DATABASE_MAP)
    ao('-e', '--engine', dest='engine', choices=engines, help='Database type, e.g. sqlite, mysql, postgresql or cockroachdb. Default is "postgresql".')
    ao('-s', '--schema', dest='schema')
    ao('-t', '--tables', dest='tables', help='Only generate the specified tables. Multiple table names should be separated by commas.')
    ao('-v', '--views', dest='views', action='store_true', help='Generate model classes for VIEWs in addition to tables.')
    ao('-i', '--info', dest='info', action='store_true', help='Add database information and other metadata to top of the generated file.')
    ao('-o', '--preserve-order', action='store_true', dest='preserve_order', help='Model definition column ordering matches source table.')
    ao('-I', '--ignore-unknown', action='store_true', dest='ignore_unknown', help='Ignore fields whose type cannot be determined.')
    ao('-L', '--legacy-naming', action='store_true', dest='legacy_naming', help='Use legacy table- and column-name generation.')
    return parser