import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
def row_gen(self):
    while True:
        rows = self.cursor.fetchmany(self.array_size)
        if not rows:
            return
        for row in rows:
            yield row