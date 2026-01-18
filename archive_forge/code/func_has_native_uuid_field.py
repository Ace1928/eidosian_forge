import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def has_native_uuid_field(self):
    is_mariadb = self.connection.mysql_is_mariadb
    return is_mariadb and self.connection.mysql_version >= (10, 7)