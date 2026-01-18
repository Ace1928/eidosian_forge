from django.core.exceptions import ImproperlyConfigured
from django.db import IntegrityError
from django.db.backends import utils as backend_utils
from django.db.backends.base.base import BaseDatabaseWrapper
from django.utils.asyncio import async_unsafe
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
from MySQLdb.constants import CLIENT, FIELD_TYPE
from MySQLdb.converters import conversions
from .client import DatabaseClient
from .creation import DatabaseCreation
from .features import DatabaseFeatures
from .introspection import DatabaseIntrospection
from .operations import DatabaseOperations
from .schema import DatabaseSchemaEditor
from .validation import DatabaseValidation
def enable_constraint_checking(self):
    """
        Re-enable foreign key checks after they have been disabled.
        """
    self.needs_rollback, needs_rollback = (False, self.needs_rollback)
    try:
        with self.cursor() as cursor:
            cursor.execute('SET foreign_key_checks=1')
    finally:
        self.needs_rollback = needs_rollback