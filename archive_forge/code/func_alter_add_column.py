from collections import namedtuple
import functools
import hashlib
import re
from peewee import *
from peewee import CommaNodeList
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import callable_
from peewee import sort_models
from peewee import sqlite3
from peewee import _truncate_constraint_name
@operation
def alter_add_column(self, table, column_name, field):
    ctx = self.make_context()
    field_null, field.null = (field.null, True)
    if field.column_name != column_name:
        field.name = field.column_name = column_name
    self._alter_table(ctx, table).literal(' ADD COLUMN ').sql(field.ddl(ctx))
    field.null = field_null
    if isinstance(field, ForeignKeyField):
        self.add_inline_fk_sql(ctx, field)
    return ctx