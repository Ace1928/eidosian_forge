import secrets
from dogpile.cache import region
from dogpile.cache import util
from oslo_cache import core as cache
from keystone.common.cache import _context_cache
import keystone.conf
def is_region_key(self, key):
    return key == self._region_key