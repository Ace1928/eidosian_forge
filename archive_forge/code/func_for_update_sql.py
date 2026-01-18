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
def for_update_sql(self, nowait=False, skip_locked=False, of=(), no_key=False):
    """
        Return the FOR UPDATE SQL clause to lock rows for an update operation.
        """
    return 'FOR%s UPDATE%s%s%s' % (' NO KEY' if no_key else '', ' OF %s' % ', '.join(of) if of else '', ' NOWAIT' if nowait else '', ' SKIP LOCKED' if skip_locked else '')