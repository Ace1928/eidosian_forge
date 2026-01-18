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
def metadef_resource_type_association_get_all_by_namespace(context, namespace_name):
    namespace = metadef_namespace_get(context, namespace_name)
    namespace_resource_types = []
    for resource_type in DATA['metadef_namespace_resource_types']:
        if resource_type['namespace_id'] == namespace['id']:
            namespace_resource_types.append(resource_type)
    return namespace_resource_types