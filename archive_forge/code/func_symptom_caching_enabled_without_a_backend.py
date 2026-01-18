from keystone.common import cache
import keystone.conf
def symptom_caching_enabled_without_a_backend():
    """Caching is not completely configured.

    Although caching is enabled in `keystone.conf [cache] enabled`, the default
    backend is still set to the no-op backend. Instead, configure keystone to
    point to a real caching backend like memcached.
    """
    return CONF.cache.enabled and CONF.cache.backend == 'dogpile.cache.null'