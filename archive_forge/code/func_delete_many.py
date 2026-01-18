import base64
import pickle
from datetime import datetime, timezone
from django.conf import settings
from django.core.cache.backends.base import DEFAULT_TIMEOUT, BaseCache
from django.db import DatabaseError, connections, models, router, transaction
from django.utils.timezone import now as tz_now
def delete_many(self, keys, version=None):
    keys = [self.make_and_validate_key(key, version=version) for key in keys]
    self._base_delete_many(keys)