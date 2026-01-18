import secrets
from dogpile.cache import region
from dogpile.cache import util
from oslo_cache import core as cache
from keystone.common.cache import _context_cache
import keystone.conf
def key_mangler(key):
    if not invalidation_manager.is_region_key(key):
        key = '%s:%s' % (key, invalidation_manager.region_id)
    if orig_key_mangler:
        key = orig_key_mangler(key)
    return key