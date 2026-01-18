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
@log_call
def image_get(context, image_id, session=None, force_show_deleted=False, v1_mode=False):
    image = copy.deepcopy(_image_get(context, image_id, force_show_deleted))
    image = _normalize_locations(context, image, force_show_deleted=force_show_deleted)
    if v1_mode:
        image = db_utils.mutate_image_dict_to_v1(image)
    return image