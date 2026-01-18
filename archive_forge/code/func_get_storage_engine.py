from collections import namedtuple
import sqlparse
from MySQLdb.constants import FIELD_TYPE
from django.db.backends.base.introspection import BaseDatabaseIntrospection
from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
from django.db.backends.base.introspection import TableInfo as BaseTableInfo
from django.db.models import Index
from django.utils.datastructures import OrderedSet
def get_storage_engine(self, cursor, table_name):
    """
        Retrieve the storage engine for a given table. Return the default
        storage engine if the table doesn't exist.
        """
    cursor.execute('\n            SELECT engine\n            FROM information_schema.tables\n            WHERE\n                table_name = %s AND\n                table_schema = DATABASE()\n            ', [table_name])
    result = cursor.fetchone()
    if not result:
        return self.connection.features._mysql_storage_engine
    return result[0]