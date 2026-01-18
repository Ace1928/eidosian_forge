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
@utils.no_4byte_params
def image_location_add(context, image_id, location):
    deleted = location['status'] in ('deleted', 'pending_delete')
    location_ref = _image_location_format(image_id, value=location['url'], meta_data=location['metadata'], status=location['status'], deleted=deleted)
    DATA['locations'].append(location_ref)
    image = DATA['images'][image_id]
    image.setdefault('locations', []).append(location_ref)