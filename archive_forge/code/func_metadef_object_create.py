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
def metadef_object_create(context, namespace_name, values):
    """Create a metadef object"""
    global DATA
    object_values = copy.deepcopy(values)
    object_name = object_values['name']
    required_attributes = ['name']
    allowed_attributes = ['name', 'description', 'json_schema', 'required']
    namespace = metadef_namespace_get(context, namespace_name)
    for object in DATA['metadef_objects']:
        if object['name'] == object_name and object['namespace_id'] == namespace['id']:
            LOG.debug('A metadata definition object with name=%(name)s in namespace=%(namespace_name)s already exists.', {'name': object_name, 'namespace_name': namespace_name})
            raise exception.MetadefDuplicateObject(object_name=object_name, namespace_name=namespace_name)
    for key in required_attributes:
        if key not in object_values:
            raise exception.Invalid('%s is a required attribute' % key)
    incorrect_keys = set(object_values.keys()) - set(allowed_attributes)
    if incorrect_keys:
        raise exception.Invalid('The keys %s are not valid' % str(incorrect_keys))
    object_values['namespace_id'] = namespace['id']
    _check_namespace_visibility(context, namespace, namespace_name)
    object = _format_object(object_values)
    DATA['metadef_objects'].append(object)
    return object