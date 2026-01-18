import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def is_sql_auto_is_null_enabled(self):
    return self.connection.mysql_server_data['sql_auto_is_null']