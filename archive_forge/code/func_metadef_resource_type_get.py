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
def metadef_resource_type_get(context, resource_type_name):
    """Get a resource type"""
    try:
        resource_type = next((resource_type for resource_type in DATA['metadef_resource_types'] if resource_type['name'] == resource_type_name))
    except StopIteration:
        LOG.debug('No resource type found with name %s', resource_type_name)
        raise exception.MetadefResourceTypeNotFound(resource_type_name=resource_type_name)
    return resource_type