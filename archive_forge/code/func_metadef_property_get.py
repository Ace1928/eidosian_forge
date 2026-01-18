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
def metadef_property_get(context, namespace_name, property_name):
    """Get a metadef property"""
    namespace = metadef_namespace_get(context, namespace_name)
    _check_namespace_visibility(context, namespace, namespace_name)
    for property in DATA['metadef_properties']:
        if property['namespace_id'] == namespace['id'] and property['name'] == property_name:
            return property
    else:
        LOG.debug('No property found with name=%(name)s in namespace=%(namespace_name)s ', {'name': property_name, 'namespace_name': namespace_name})
        raise exception.MetadefPropertyNotFound(namespace_name=namespace_name, property_name=property_name)