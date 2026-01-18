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
def cache_key_culling_sql(self):
    """
        Return an SQL query that retrieves the first cache key greater than the
        n smallest.

        This is used by the 'db' cache backend to determine where to start
        culling.
        """
    cache_key = self.quote_name('cache_key')
    return f'SELECT {cache_key} FROM %s ORDER BY {cache_key} LIMIT 1 OFFSET %%s'