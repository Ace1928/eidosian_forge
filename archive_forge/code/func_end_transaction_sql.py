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
def end_transaction_sql(self, success=True):
    """Return the SQL statement required to end a transaction."""
    if not success:
        return 'ROLLBACK;'
    return 'COMMIT;'