from oslo_config import cfg
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_log import log as logging
from oslo_utils import units
from glance.common import exception
from glance.db.sqlalchemy import api as db
from glance.i18n import _LE
def enforce_image_count_uploading(context, project_id):
    """Enforce the image_count_uploading quota.

    This enforces the total count of images in any state of upload by
    the supplied project_id.

    :param delta: This defaults to one, but should be zero when checking
                  an operation on an image that already counts against this
                  quota (i.e. a stage operation of an existing queue image).
    """
    _enforce_one(context, project_id, QUOTA_IMAGE_COUNT_UPLOADING, lambda: db.user_get_uploading_count(context, project_id), delta=0)