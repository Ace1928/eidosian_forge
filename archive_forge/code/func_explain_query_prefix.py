import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def explain_query_prefix(self, format=None, **options):
    if format and format.upper() == 'TEXT':
        format = 'TRADITIONAL'
    elif not format and 'TREE' in self.connection.features.supported_explain_formats:
        format = 'TREE'
    analyze = options.pop('analyze', False)
    prefix = super().explain_query_prefix(format, **options)
    if analyze and self.connection.features.supports_explain_analyze:
        prefix = 'ANALYZE' if self.connection.mysql_is_mariadb else prefix + ' ANALYZE'
    if format and (not (analyze and (not self.connection.mysql_is_mariadb))):
        prefix += ' FORMAT=%s' % format
    return prefix