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
def image_destroy(context, image_id):
    global DATA
    try:
        delete_time = timeutils.utcnow()
        DATA['images'][image_id]['deleted'] = True
        DATA['images'][image_id]['deleted_at'] = delete_time
        if DATA['images'][image_id]['status'] not in ['deleted', 'pending_delete']:
            DATA['images'][image_id]['status'] = 'deleted'
        _image_locations_delete_all(context, image_id, delete_time=delete_time)
        for prop in DATA['images'][image_id]['properties']:
            image_property_delete(context, prop['name'], image_id)
        members = image_member_find(context, image_id=image_id)
        for member in members:
            image_member_delete(context, member['id'])
        tags = image_tag_get_all(context, image_id)
        for tag in tags:
            image_tag_delete(context, image_id, tag)
        return _normalize_locations(context, copy.deepcopy(DATA['images'][image_id]))
    except KeyError:
        raise exception.ImageNotFound()