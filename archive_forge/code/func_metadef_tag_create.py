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
def metadef_tag_create(context, namespace_name, values):
    """Create a metadef tag"""
    global DATA
    tag_values = copy.deepcopy(values)
    tag_name = tag_values['name']
    required_attributes = ['name']
    allowed_attributes = ['name']
    namespace = metadef_namespace_get(context, namespace_name)
    for tag in DATA['metadef_tags']:
        if tag['name'] == tag_name and tag['namespace_id'] == namespace['id']:
            LOG.debug('A metadata definition tag with name=%(name)s in namespace=%(namespace_name)s already exists.', {'name': tag_name, 'namespace_name': namespace_name})
            raise exception.MetadefDuplicateTag(name=tag_name, namespace_name=namespace_name)
    for key in required_attributes:
        if key not in tag_values:
            raise exception.Invalid('%s is a required attribute' % key)
    incorrect_keys = set(tag_values.keys()) - set(allowed_attributes)
    if incorrect_keys:
        raise exception.Invalid('The keys %s are not valid' % str(incorrect_keys))
    tag_values['namespace_id'] = namespace['id']
    _check_namespace_visibility(context, namespace, namespace_name)
    tag = _format_tag(tag_values)
    DATA['metadef_tags'].append(tag)
    return tag