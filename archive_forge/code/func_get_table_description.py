from collections import namedtuple
import sqlparse
from MySQLdb.constants import FIELD_TYPE
from django.db.backends.base.introspection import BaseDatabaseIntrospection
from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
from django.db.backends.base.introspection import TableInfo as BaseTableInfo
from django.db.models import Index
from django.utils.datastructures import OrderedSet
def get_table_description(self, cursor, table_name):
    """
        Return a description of the table with the DB-API cursor.description
        interface."
        """
    json_constraints = {}
    if self.connection.mysql_is_mariadb and self.connection.features.can_introspect_json_field:
        cursor.execute("\n                SELECT c.constraint_name AS column_name\n                FROM information_schema.check_constraints AS c\n                WHERE\n                    c.table_name = %s AND\n                    LOWER(c.check_clause) =\n                        'json_valid(`' + LOWER(c.constraint_name) + '`)' AND\n                    c.constraint_schema = DATABASE()\n                ", [table_name])
        json_constraints = {row[0] for row in cursor.fetchall()}
    cursor.execute('\n            SELECT  table_collation\n            FROM    information_schema.tables\n            WHERE   table_schema = DATABASE()\n            AND     table_name = %s\n            ', [table_name])
    row = cursor.fetchone()
    default_column_collation = row[0] if row else ''
    cursor.execute("\n            SELECT\n                column_name, data_type, character_maximum_length,\n                numeric_precision, numeric_scale, extra, column_default,\n                CASE\n                    WHEN collation_name = %s THEN NULL\n                    ELSE collation_name\n                END AS collation_name,\n                CASE\n                    WHEN column_type LIKE '%% unsigned' THEN 1\n                    ELSE 0\n                END AS is_unsigned,\n                column_comment\n            FROM information_schema.columns\n            WHERE table_name = %s AND table_schema = DATABASE()\n            ", [default_column_collation, table_name])
    field_info = {line[0]: InfoLine(*line) for line in cursor.fetchall()}
    cursor.execute('SELECT * FROM %s LIMIT 1' % self.connection.ops.quote_name(table_name))

    def to_int(i):
        return int(i) if i is not None else i
    fields = []
    for line in cursor.description:
        info = field_info[line[0]]
        fields.append(FieldInfo(*line[:2], to_int(info.max_len) or line[2], to_int(info.max_len) or line[3], to_int(info.num_prec) or line[4], to_int(info.num_scale) or line[5], line[6], info.column_default, info.collation, info.extra, info.is_unsigned, line[0] in json_constraints, info.comment, info.data_type))
    return fields