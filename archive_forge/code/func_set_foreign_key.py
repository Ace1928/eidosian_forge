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
def set_foreign_key(self, foreign_key, model_names, dest=None, related_name=None):
    self.foreign_key = foreign_key
    self.field_class = ForeignKeyField
    if foreign_key.dest_table == foreign_key.table:
        self.rel_model = "'self'"
    else:
        self.rel_model = model_names[foreign_key.dest_table]
    self.to_field = dest and dest.name or None
    self.related_name = related_name or None