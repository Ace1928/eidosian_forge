import secrets
from dogpile.cache import region
from dogpile.cache import util
from oslo_cache import core as cache
from keystone.common.cache import _context_cache
import keystone.conf
def get_memoization_decorator(group, expiration_group=None, region=None):
    if region is None:
        region = CACHE_REGION
    return cache.get_memoization_decorator(CONF, region, group, expiration_group=expiration_group)