import secrets
from dogpile.cache import region
from dogpile.cache import util
from oslo_cache import core as cache
from keystone.common.cache import _context_cache
import keystone.conf
@property
def region_id(self):
    return self._invalidation_region.get_or_create(self._region_key, self._generate_new_id, expiration_time=-1)