import operator
from django.db.backends.base.features import BaseDatabaseFeatures
from django.utils.functional import cached_property
@cached_property
def has_zoneinfo_database(self):
    return self.connection.mysql_server_data['has_zoneinfo_database']