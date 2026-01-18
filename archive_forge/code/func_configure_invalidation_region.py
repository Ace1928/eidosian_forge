import secrets
from dogpile.cache import region
from dogpile.cache import util
from oslo_cache import core as cache
from keystone.common.cache import _context_cache
import keystone.conf
def configure_invalidation_region():
    if CACHE_INVALIDATION_REGION.is_configured:
        return
    config_dict = cache._build_cache_config(CONF)
    config_dict['expiration_time'] = None
    CACHE_INVALIDATION_REGION.configure_from_config(config_dict, '%s.' % CONF.cache.config_prefix)
    CACHE_INVALIDATION_REGION.wrap(_context_cache._ResponseCacheProxy)
    if CACHE_INVALIDATION_REGION.key_mangler is None:
        CACHE_INVALIDATION_REGION.key_mangler = _sha1_mangle_key