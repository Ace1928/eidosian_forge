import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def can_rename_index(self):
    if self.connection.mysql_is_mariadb:
        return self.connection.mysql_version >= (10, 5, 2)
    return True