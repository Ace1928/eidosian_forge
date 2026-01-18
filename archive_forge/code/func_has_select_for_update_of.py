import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def has_select_for_update_of(self):
    return not self.connection.mysql_is_mariadb