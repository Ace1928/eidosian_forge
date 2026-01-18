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
def apply_default(self, table, column_name, field):
    default = field.default
    if callable_(default):
        default = default()
    return self.make_context().literal('UPDATE ').sql(Entity(table)).literal(' SET ').sql(Expression(Entity(column_name), OP.EQ, field.db_value(default), flat=True))