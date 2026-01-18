import secrets
from dogpile.cache import region
from dogpile.cache import util
from oslo_cache import core as cache
from keystone.common.cache import _context_cache
import keystone.conf
def create_region(name):
    """Create a dopile region.

    Wraps oslo_cache.core.create_region. This is used to ensure that the
    Region is properly patched and allows us to more easily specify a region
    name.

    :param str name: The region name
    :returns: The new region.
    :rtype: :class:`dogpile.cache.region.CacheRegion`

    """
    region = cache.create_region()
    region.name = name
    return region