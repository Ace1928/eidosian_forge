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
def snippet(self, left, right, over_length='...', max_tokens=16):
    if not 0 < max_tokens < 65:
        raise ValueError('max_tokens must be between 1 and 64 (inclusive)')
    column_idx = self.fts_column_index
    return fn.snippet(self.model._meta.entity, column_idx, left, right, over_length, max_tokens)