import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def supports_index_column_ordering(self):
    if self._mysql_storage_engine != 'InnoDB':
        return False
    if self.connection.mysql_is_mariadb:
        return self.connection.mysql_version >= (10, 8)
    return True