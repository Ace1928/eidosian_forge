import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def ignores_table_name_case(self):
    return self.connection.mysql_server_data['lower_case_table_names']