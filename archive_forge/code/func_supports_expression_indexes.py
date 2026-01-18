import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def supports_expression_indexes(self):
    return not self.connection.mysql_is_mariadb and self._mysql_storage_engine != 'MyISAM' and (self.connection.mysql_version >= (8, 0, 13))