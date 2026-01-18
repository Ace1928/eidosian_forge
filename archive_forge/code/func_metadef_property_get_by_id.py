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
def metadef_property_get_by_id(context, namespace_name, property_id):
    """Get a metadef property"""
    namespace = metadef_namespace_get(context, namespace_name)
    _check_namespace_visibility(context, namespace, namespace_name)
    for property in DATA['metadef_properties']:
        if property['namespace_id'] == namespace['id'] and property['id'] == property_id:
            return property
    else:
        msg = _('Metadata definition property not found for id=%s') % property_id
        LOG.warning(msg)
        raise exception.MetadefPropertyNotFound(msg)