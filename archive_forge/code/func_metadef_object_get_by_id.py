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
def metadef_object_get_by_id(context, namespace_name, object_id):
    """Get a metadef object"""
    namespace = metadef_namespace_get(context, namespace_name)
    _check_namespace_visibility(context, namespace, namespace_name)
    for object in DATA['metadef_objects']:
        if object['namespace_id'] == namespace['id'] and object['id'] == object_id:
            return object
    else:
        msg = _('Metadata definition object not found for id=%s') % object_id
        LOG.warning(msg)
        raise exception.MetadefObjectNotFound(msg)