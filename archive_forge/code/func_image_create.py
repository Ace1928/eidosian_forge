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
def image_create(context, image_values, v1_mode=False):
    global DATA
    image_id = image_values.get('id', str(uuid.uuid4()))
    if image_id in DATA['images']:
        raise exception.Duplicate()
    if 'status' not in image_values:
        raise exception.Invalid('status is a required attribute')
    allowed_keys = set(['id', 'name', 'status', 'min_ram', 'min_disk', 'size', 'virtual_size', 'checksum', 'locations', 'owner', 'protected', 'is_public', 'container_format', 'disk_format', 'created_at', 'updated_at', 'deleted', 'deleted_at', 'properties', 'tags', 'visibility', 'os_hidden', 'os_hash_algo', 'os_hash_value'])
    incorrect_keys = set(image_values.keys()) - allowed_keys
    if incorrect_keys:
        raise exception.Invalid('The keys %s are not valid' % str(incorrect_keys))
    image = _image_format(image_id, **image_values)
    DATA['images'][image_id] = image
    DATA['tags'][image_id] = image.pop('tags', [])
    image = _normalize_locations(context, copy.deepcopy(image))
    if v1_mode:
        image = db_utils.mutate_image_dict_to_v1(image)
    return image