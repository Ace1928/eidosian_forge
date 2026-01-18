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
def clean_options(cls, options):
    filename = cls._meta.filename
    if not filename:
        raise ValueError('LSM1 extension requires that you specify a filename for the LSM database.')
    elif len(filename) >= 2 and filename[0] != '"':
        filename = '"%s"' % filename
    if not cls._meta.primary_key:
        raise ValueError('LSM1 models must specify a primary-key field.')
    key = cls._meta.primary_key
    if isinstance(key, AutoField):
        raise ValueError('LSM1 models must explicitly declare a primary key field.')
    if not isinstance(key, (TextField, BlobField, IntegerField)):
        raise ValueError('LSM1 key must be a TextField, BlobField, or IntegerField.')
    key._hidden = True
    if isinstance(key, IntegerField):
        data_type = 'UINT'
    elif isinstance(key, BlobField):
        data_type = 'BLOB'
    else:
        data_type = 'TEXT'
    cls._meta.prefix_arguments = [filename, '"%s"' % key.name, data_type]
    if len(cls._meta.sorted_fields) == 2:
        cls._meta._value_field = cls._meta.sorted_fields[1]
    else:
        cls._meta._value_field = None
    return options