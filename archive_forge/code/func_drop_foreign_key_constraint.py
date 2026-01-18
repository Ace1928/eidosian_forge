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
def drop_foreign_key_constraint(self, table, column_name):
    fk_constraint = self.get_foreign_key_constraint(table, column_name)
    return self._alter_table(self.make_context(), table).literal(' DROP FOREIGN KEY ').sql(Entity(fk_constraint))