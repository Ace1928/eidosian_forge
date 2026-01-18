import re
import time
from django.core.cache.backends.base import (
from django.utils.functional import cached_property
def get_backend_timeout(self, timeout=DEFAULT_TIMEOUT):
    """
        Memcached deals with long (> 30 days) timeouts in a special
        way. Call this function to obtain a safe value for your timeout.
        """
    if timeout == DEFAULT_TIMEOUT:
        timeout = self.default_timeout
    if timeout is None:
        return 0
    elif int(timeout) == 0:
        timeout = -1
    if timeout > 2592000:
        timeout += int(time.time())
    return int(timeout)