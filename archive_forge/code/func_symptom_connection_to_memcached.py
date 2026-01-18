from keystone.common import cache
import keystone.conf
def symptom_connection_to_memcached():
    """Memcached isn't reachable.

    Caching is enabled and the `keystone.conf [cache] backend` option is
    configured but one or more Memcached servers are not reachable or marked
    as dead. Please ensure `keystone.conf [cache] memcache_servers` is
    configured properly.
    """
    memcached_drivers = ['dogpile.cache.memcached', 'oslo_cache.memcache_pool']
    if CONF.cache.enabled and CONF.cache.backend in memcached_drivers:
        cache.configure_cache()
        cache_stats = cache.CACHE_REGION.actual_backend.client.get_stats()
        memcached_server_count = len(CONF.cache.memcache_servers)
        if len(cache_stats) != memcached_server_count:
            return True
        else:
            return False
    else:
        return False