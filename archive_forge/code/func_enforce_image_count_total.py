from oslo_config import cfg
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_log import log as logging
from oslo_utils import units
from glance.common import exception
from glance.db.sqlalchemy import api as db
from glance.i18n import _LE
def enforce_image_count_total(context, project_id):
    """Enforce the image_count_total quota.

    This enforces the total count of non-deleted images owned by the
    supplied project_id.
    """
    _enforce_one(context, project_id, QUOTA_IMAGE_COUNT_TOTAL, lambda: db.user_get_image_count(context, project_id), delta=1)