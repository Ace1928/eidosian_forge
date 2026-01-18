import operator
from peewee import *
from peewee import sqlite3
from peewee import Expression
from playhouse.fields import PickleField
def convert_expression(self, expr):
    if not isinstance(expr, Expression):
        return (self.key == expr, True)
    return (expr, False)