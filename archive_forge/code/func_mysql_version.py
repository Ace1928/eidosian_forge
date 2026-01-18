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
@cached_property
def mysql_version(self):
    match = server_version_re.match(self.mysql_server_info)
    if not match:
        raise Exception('Unable to determine MySQL version from version string %r' % self.mysql_server_info)
    return tuple((int(x) for x in match.groups()))