import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def return_insert_columns(self, fields):
    if not fields:
        return ('', ())
    columns = ['%s.%s' % (self.quote_name(field.model._meta.db_table), self.quote_name(field.column)) for field in fields]
    return ('RETURNING %s' % ', '.join(columns), ())