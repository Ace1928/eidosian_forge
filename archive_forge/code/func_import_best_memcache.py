from kombu.utils.encoding import bytes_to_str, ensure_bytes
from kombu.utils.objects import cached_property
from celery.exceptions import ImproperlyConfigured
from celery.utils.functional import LRUCache
from .base import KeyValueStoreBackend
def import_best_memcache():
    if _imp[0] is None:
        is_pylibmc, memcache_key_t = (False, bytes_to_str)
        try:
            import pylibmc as memcache
            is_pylibmc = True
        except ImportError:
            try:
                import memcache
            except ImportError:
                raise ImproperlyConfigured(REQUIRES_BACKEND)
        _imp[0] = (is_pylibmc, memcache, memcache_key_t)
    return _imp[0]