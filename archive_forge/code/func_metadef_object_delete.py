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
def metadef_object_delete(context, namespace_name, object_name):
    """Delete a metadef object"""
    global DATA
    object = metadef_object_get(context, namespace_name, object_name)
    DATA['metadef_objects'].remove(object)
    return object