import json
import math
import re
import struct
import sys
from peewee import *
from peewee import ColumnBase
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Insert
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import VirtualField
from peewee import merge_dict
from peewee import sqlite3
def on_rollback(self, fn):
    self._rollback_hook = fn
    if not self.is_closed():
        self._conn_helper.set_rollback_hook(fn)
    return fn