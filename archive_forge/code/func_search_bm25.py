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
def search_bm25(cls, term, weights=None, with_score=False, score_alias='score', explicit_ordering=False):
    """Full-text search using selected `term`."""
    if not weights:
        rank = SQL('rank')
    elif isinstance(weights, dict):
        weight_args = []
        for field in cls._meta.sorted_fields:
            if isinstance(field, SearchField) and (not field.unindexed):
                weight_args.append(weights.get(field, weights.get(field.name, 1.0)))
        rank = fn.bm25(cls._meta.entity, *weight_args)
    else:
        rank = fn.bm25(cls._meta.entity, *weights)
    selection = ()
    order_by = rank
    if with_score:
        selection = (cls, rank.alias(score_alias))
    if with_score and (not explicit_ordering):
        order_by = SQL(score_alias)
    return cls.select(*selection).where(cls.match(FTS5Model.clean_query(term))).order_by(order_by)