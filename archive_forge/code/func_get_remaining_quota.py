import re
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import units
import glance.async_
from glance.common import exception
from glance.i18n import _, _LE, _LW
def get_remaining_quota(context, db_api, image_id=None):
    """Method called to see if the user is allowed to store an image.

    Checks if it is allowed based on the given size in glance based on their
    quota and current usage.

    :param context:
    :param db_api:  The db_api in use for this configuration
    :param image_id: The image that will be replaced with this new data size
    :returns: The number of bytes the user has remaining under their quota.
             None means infinity
    """
    users_quota = CONF.user_storage_quota
    pattern = re.compile('^(\\d+)((K|M|G|T)?B)?$')
    match = pattern.match(users_quota)
    if not match:
        LOG.error(_LE('Invalid value for option user_storage_quota: %(users_quota)s'), {'users_quota': users_quota})
        raise exception.InvalidOptionValue(option='user_storage_quota', value=users_quota)
    quota_value, quota_unit = match.groups()[0:2]
    quota_unit = quota_unit or 'B'
    factor = getattr(units, quota_unit.replace('B', 'i'), 1)
    users_quota = int(quota_value) * factor
    if users_quota <= 0:
        return
    usage = db_api.user_get_storage_usage(context, context.owner, image_id=image_id)
    return users_quota - usage