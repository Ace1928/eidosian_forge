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
def metadef_property_create(context, namespace_name, values):
    """Create a metadef property"""
    global DATA
    property_values = copy.deepcopy(values)
    property_name = property_values['name']
    required_attributes = ['name']
    allowed_attributes = ['name', 'description', 'json_schema', 'required']
    namespace = metadef_namespace_get(context, namespace_name)
    for property in DATA['metadef_properties']:
        if property['name'] == property_name and property['namespace_id'] == namespace['id']:
            LOG.debug('Can not create metadata definition property. A property with name=%(name)s already exists in namespace=%(namespace_name)s.', {'name': property_name, 'namespace_name': namespace_name})
            raise exception.MetadefDuplicateProperty(property_name=property_name, namespace_name=namespace_name)
    for key in required_attributes:
        if key not in property_values:
            raise exception.Invalid('%s is a required attribute' % key)
    incorrect_keys = set(property_values.keys()) - set(allowed_attributes)
    if incorrect_keys:
        raise exception.Invalid('The keys %s are not valid' % str(incorrect_keys))
    property_values['namespace_id'] = namespace['id']
    _check_namespace_visibility(context, namespace, namespace_name)
    property = _format_property(property_values)
    DATA['metadef_properties'].append(property)
    return property