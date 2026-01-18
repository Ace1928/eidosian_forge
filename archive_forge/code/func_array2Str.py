from decimal import Decimal
from MySQLdb._mysql import string_literal
from MySQLdb.constants import FIELD_TYPE, FLAG
from MySQLdb.times import (
from MySQLdb._exceptions import ProgrammingError
import array
def array2Str(o, d):
    return Thing2Literal(o.tostring(), d)