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
def drop_not_null(self, table, column):

    def _drop_not_null(column_name, column_def):
        return column_def.replace('NOT NULL', '')
    return self._update_column(table, column, _drop_not_null)