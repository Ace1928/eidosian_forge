from django.contrib.gis.gdal import OGRGeomType
from django.db.backends.sqlite3.introspection import (
def get_constraints(self, cursor, table_name):
    constraints = super().get_constraints(cursor, table_name)
    cursor.execute('SELECT f_geometry_column FROM geometry_columns WHERE f_table_name=%s AND spatial_index_enabled=1', (table_name,))
    for row in cursor.fetchall():
        constraints['%s__spatial__index' % row[0]] = {'columns': [row[0]], 'primary_key': False, 'unique': False, 'foreign_key': None, 'check': False, 'index': True}
    return constraints