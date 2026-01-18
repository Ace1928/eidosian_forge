import operator
from django.db import transaction
from django.db.backends.base.features import BaseDatabaseFeatures
from django.db.utils import OperationalError
from django.utils.functional import cached_property
from .base import Database
@cached_property
def supports_json_field(self):
    with self.connection.cursor() as cursor:
        try:
            with transaction.atomic(self.connection.alias):
                cursor.execute('SELECT JSON(\'{"a": "b"}\')')
        except OperationalError:
            return False
    return True