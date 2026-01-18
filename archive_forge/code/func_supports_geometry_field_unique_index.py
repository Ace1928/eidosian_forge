from django.contrib.gis.db.backends.base.features import BaseSpatialFeatures
from django.db.backends.mysql.features import DatabaseFeatures as MySQLDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def supports_geometry_field_unique_index(self):
    return self.connection.mysql_is_mariadb