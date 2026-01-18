from oslo_config import cfg
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_log import log as logging
from oslo_utils import units
from glance.common import exception
from glance.db.sqlalchemy import api as db
from glance.i18n import _LE
def enforce_image_size_total(context, project_id, delta=0):
    """Enforce the image_size_total quota.

    This enforces the total image size quota for the supplied project_id.
    """
    _enforce_one(context, project_id, QUOTA_IMAGE_SIZE_TOTAL, lambda: db.user_get_storage_usage(context, project_id) // units.Mi, delta=delta)