import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def minimum_database_version(self):
    if self.connection.mysql_is_mariadb:
        return (10, 4)
    else:
        return (8, 0, 11)