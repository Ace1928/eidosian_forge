import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def supports_explain_analyze(self):
    return self.connection.mysql_is_mariadb or self.connection.mysql_version >= (8, 0, 18)