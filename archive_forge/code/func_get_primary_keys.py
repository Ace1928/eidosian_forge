from collections import namedtuple
from inspect import isclass
import re
import warnings
from peewee import *
from peewee import _StringField
from peewee import _query_val_transform
from peewee import CommaNodeList
from peewee import SCOPE_VALUES
from peewee import make_snake_case
from peewee import text_type
def get_primary_keys(self, table, schema=None):
    schema = schema or 'public'
    return super(PostgresqlMetadata, self).get_primary_keys(table, schema)