import glance_store
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from glance.api import common as api_common
from glance.common import exception
from glance import context
from glance.i18n import _LI, _LW
from glance.image_cache import base
def fetch_image_into_cache(self, image_id):
    ctx = context.RequestContext(is_admin=True, show_deleted=True, roles=['admin'])
    try:
        image_repo = self.gateway.get_repo(ctx)
        image = image_repo.get(image_id)
    except exception.NotFound:
        LOG.warning(_LW("Image '%s' not found"), image_id)
        return False
    if image.status != 'active':
        LOG.warning(_LW("Image '%s' is not active. Not caching."), image_id)
        return False
    for loc in image.locations:
        if CONF.enabled_backends:
            image_data, image_size = glance_store.get(loc['url'], None, context=ctx)
        else:
            image_data, image_size = glance_store.get_from_backend(loc['url'], context=ctx)
        LOG.debug("Caching image '%s'", image_id)
        cache_tee_iter = self.cache.cache_tee_iter(image_id, image_data, image.checksum)
        list(cache_tee_iter)
        return True