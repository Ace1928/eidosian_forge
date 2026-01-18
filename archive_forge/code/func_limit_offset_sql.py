import datetime
import decimal
import json
import warnings
from importlib import import_module
import sqlparse
from django.conf import settings
from django.db import NotSupportedError, transaction
from django.db.backends import utils
from django.db.models.expressions import Col
from django.utils import timezone
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.encoding import force_str
def limit_offset_sql(self, low_mark, high_mark):
    """Return LIMIT/OFFSET SQL clause."""
    limit, offset = self._get_limit_offset_params(low_mark, high_mark)
    return ' '.join((sql for sql in ('LIMIT %d' % limit if limit else None, 'OFFSET %d' % offset if offset else None) if sql))