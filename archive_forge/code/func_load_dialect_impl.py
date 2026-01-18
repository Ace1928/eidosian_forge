from oslo_serialization import jsonutils
from sqlalchemy.dialects import mysql
from sqlalchemy import types
def load_dialect_impl(self, dialect):
    if dialect.name == 'mysql':
        return dialect.type_descriptor(mysql.LONGTEXT())
    else:
        return self.impl