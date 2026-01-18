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
def metadef_object_get(context, namespace_name, object_name):
    """Get a metadef object"""
    namespace = metadef_namespace_get(context, namespace_name)
    _check_namespace_visibility(context, namespace, namespace_name)
    for object in DATA['metadef_objects']:
        if object['namespace_id'] == namespace['id'] and object['name'] == object_name:
            return object
    else:
        LOG.debug('The metadata definition object with name=%(name)s was not found in namespace=%(namespace_name)s.', {'name': object_name, 'namespace_name': namespace_name})
        raise exception.MetadefObjectNotFound(namespace_name=namespace_name, object_name=object_name)