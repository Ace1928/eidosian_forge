import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def sequence_reset_by_name_sql(self, style, sequences):
    return ['%s %s %s %s = 1;' % (style.SQL_KEYWORD('ALTER'), style.SQL_KEYWORD('TABLE'), style.SQL_FIELD(self.quote_name(sequence_info['table'])), style.SQL_FIELD('AUTO_INCREMENT')) for sequence_info in sequences]