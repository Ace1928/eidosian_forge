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
@staticmethod
def clean_query(query, replace=chr(26)):
    """
        Clean a query of invalid tokens.
        """
    accum = []
    any_invalid = False
    tokens = _quote_re.findall(query)
    for token in tokens:
        if token.startswith('"') and token.endswith('"'):
            accum.append(token)
            continue
        token_set = set(token)
        invalid_for_token = token_set & _invalid_ascii
        if invalid_for_token:
            any_invalid = True
            for c in invalid_for_token:
                token = token.replace(c, replace)
        accum.append(token)
    if any_invalid:
        return ' '.join(accum)
    return query