import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def on_conflict_suffix_sql(self, fields, on_conflict, update_fields, unique_fields):
    if on_conflict == OnConflict.UPDATE:
        conflict_suffix_sql = 'ON DUPLICATE KEY UPDATE %(fields)s'
        if not self.connection.mysql_is_mariadb:
            if self.connection.mysql_version >= (8, 0, 19):
                conflict_suffix_sql = f'AS new {conflict_suffix_sql}'
                field_sql = '%(field)s = new.%(field)s'
            else:
                field_sql = '%(field)s = VALUES(%(field)s)'
        else:
            field_sql = '%(field)s = VALUE(%(field)s)'
        fields = ', '.join([field_sql % {'field': field} for field in map(self.quote_name, update_fields)])
        return conflict_suffix_sql % {'fields': fields}
    return super().on_conflict_suffix_sql(fields, on_conflict, update_fields, unique_fields)