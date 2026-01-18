from collections import namedtuple
import sqlparse
from django.db import DatabaseError
from django.db.backends.base.introspection import BaseDatabaseIntrospection
from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
from django.db.backends.base.introspection import TableInfo
from django.db.models import Index
from django.utils.regex_helper import _lazy_re_compile
def get_field_size(name):
    """Extract the size number from a "varchar(11)" type name"""
    m = field_size_re.search(name)
    return int(m[1]) if m else None