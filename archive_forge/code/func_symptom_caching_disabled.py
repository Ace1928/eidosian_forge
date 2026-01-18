from keystone.common import cache
import keystone.conf
def symptom_caching_disabled():
    """`keystone.conf [cache] enabled` is not enabled.

    Caching greatly improves the performance of keystone, and it is highly
    recommended that you enable it.
    """
    return not CONF.cache.enabled