import copy
import functools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db import utils as db_utils
from glance.i18n import _, _LI, _LW
def is_image_visible(context, image, status=None):
    if status == 'all':
        status = None
    return db_utils.is_image_visible(context, image, image_member_find, status)