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
def metadef_tag_get(context, namespace_name, name):
    """Get a metadef tag"""
    namespace = metadef_namespace_get(context, namespace_name)
    _check_namespace_visibility(context, namespace, namespace_name)
    for tag in DATA['metadef_tags']:
        if tag['namespace_id'] == namespace['id'] and tag['name'] == name:
            return tag
    else:
        LOG.debug('The metadata definition tag with name=%(name)s was not found in namespace=%(namespace_name)s.', {'name': name, 'namespace_name': namespace_name})
        raise exception.MetadefTagNotFound(name=name, namespace_name=namespace_name)