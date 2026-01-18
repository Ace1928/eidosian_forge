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
def image_property_delete(context, prop_ref, image_ref):
    prop = None
    for p in DATA['images'][image_ref]['properties']:
        if p['name'] == prop_ref:
            prop = p
    if not prop:
        raise exception.NotFound()
    prop['deleted_at'] = timeutils.utcnow()
    prop['deleted'] = True
    return prop