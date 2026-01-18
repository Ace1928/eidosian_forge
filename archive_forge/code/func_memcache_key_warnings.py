import time
import warnings
from asgiref.sync import sync_to_async
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
def memcache_key_warnings(key):
    if len(key) > MEMCACHE_MAX_KEY_LENGTH:
        yield ('Cache key will cause errors if used with memcached: %r (longer than %s)' % (key, MEMCACHE_MAX_KEY_LENGTH))
    if memcached_error_chars_re.search(key):
        yield f'Cache key contains characters that will cause errors if used with memcached: {key!r}'