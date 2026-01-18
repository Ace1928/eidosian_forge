import re
import time
from django.core.cache.backends.base import (
from django.utils.functional import cached_property
def set_many(self, data, timeout=DEFAULT_TIMEOUT, version=None):
    safe_data = {}
    original_keys = {}
    for key, value in data.items():
        safe_key = self.make_and_validate_key(key, version=version)
        safe_data[safe_key] = value
        original_keys[safe_key] = key
    failed_keys = self._cache.set_multi(safe_data, self.get_backend_timeout(timeout))
    return [original_keys[k] for k in failed_keys]