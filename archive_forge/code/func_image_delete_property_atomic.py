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
def image_delete_property_atomic(image_id, name, value):
    try:
        image = DATA['images'][image_id]
    except KeyError:
        LOG.warning(_LW('Could not find image %s'), image_id)
        raise exception.ImageNotFound()
    for i, prop in enumerate(image['properties']):
        if prop['name'] == name and prop['value'] == value:
            del image['properties'][i]
            return
    raise exception.NotFound()