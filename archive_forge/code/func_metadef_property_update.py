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
def metadef_property_update(context, namespace_name, property_id, values):
    """Update a metadef property"""
    global DATA
    namespace = metadef_namespace_get(context, namespace_name)
    _check_namespace_visibility(context, namespace, namespace_name)
    property = metadef_property_get_by_id(context, namespace_name, property_id)
    if property['name'] != values['name']:
        for db_property in DATA['metadef_properties']:
            if db_property['name'] == values['name'] and db_property['namespace_id'] == namespace['id']:
                LOG.debug('Invalid update. It would result in a duplicate metadata definition property with the same name=%(name)s in namespace=%(namespace_name)s.', {'name': property['name'], 'namespace_name': namespace_name})
                emsg = _('Invalid update. It would result in a duplicate metadata definition property with the same name=%(name)s in namespace=%(namespace_name)s.') % {'name': property['name'], 'namespace_name': namespace_name}
                raise exception.MetadefDuplicateProperty(emsg)
    DATA['metadef_properties'].remove(property)
    property.update(values)
    property['updated_at'] = timeutils.utcnow()
    DATA['metadef_properties'].append(property)
    return property