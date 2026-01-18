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
def is_image_cached_for_node(context, node_reference_url, image_id):
    global DATA
    node_reference = node_reference_get_by_url(context, node_reference_url)
    all_images = DATA['cached_images']
    for image_id in all_images:
        if all_images[image_id]['node_reference_id'] == node_reference['node_reference_id'] and image_id == all_images[image_id]['image_id']:
            return True
    return False