import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
def delete_endpoints(self, **kwargs):
    cache_key = self._create_cache_key(**kwargs)
    if cache_key in self._cache:
        del self._cache[cache_key]