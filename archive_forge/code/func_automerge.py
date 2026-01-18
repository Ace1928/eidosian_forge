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
@classmethod
def automerge(cls, level):
    if not 0 <= level <= 16:
        raise ValueError('level must be between 0 and 16')
    return cls._fts_cmd('automerge', rank=level)