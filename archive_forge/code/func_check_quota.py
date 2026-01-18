import re
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import units
import glance.async_
from glance.common import exception
from glance.i18n import _, _LE, _LW
def check_quota(context, image_size, db_api, image_id=None):
    """Method called to see if the user is allowed to store an image.

    Checks if it is allowed based on the given size in glance based on their
    quota and current usage.

    :param context:
    :param image_size:  The size of the image we hope to store
    :param db_api:  The db_api in use for this configuration
    :param image_id: The image that will be replaced with this new data size
    :returns:
    """
    if CONF.use_keystone_limits:
        return
    remaining = get_remaining_quota(context, db_api, image_id=image_id)
    if remaining is None:
        return
    user = getattr(context, 'user_id', '<unknown>')
    if image_size is None:
        if remaining <= 0:
            LOG.warning(_LW('User %(user)s attempted to upload an image of unknown size that will exceed the quota. %(remaining)d bytes remaining.'), {'user': user, 'remaining': remaining})
            raise exception.StorageQuotaFull(image_size=image_size, remaining=remaining)
        return
    if image_size > remaining:
        LOG.warning(_LW('User %(user)s attempted to upload an image of size %(size)d that will exceed the quota. %(remaining)d bytes remaining.'), {'user': user, 'size': image_size, 'remaining': remaining})
        raise exception.StorageQuotaFull(image_size=image_size, remaining=remaining)
    return remaining