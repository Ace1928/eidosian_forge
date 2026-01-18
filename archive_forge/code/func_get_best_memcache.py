from kombu.utils.encoding import bytes_to_str, ensure_bytes
from kombu.utils.objects import cached_property
from celery.exceptions import ImproperlyConfigured
from celery.utils.functional import LRUCache
from .base import KeyValueStoreBackend
def get_best_memcache(*args, **kwargs):
    is_pylibmc, memcache, key_t = import_best_memcache()
    Client = _Client = memcache.Client
    if not is_pylibmc:

        def Client(*args, **kwargs):
            kwargs.pop('behaviors', None)
            return _Client(*args, **kwargs)
    return (Client, key_t)