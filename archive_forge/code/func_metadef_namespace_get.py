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
def metadef_namespace_get(context, namespace_name):
    """Get a namespace object"""
    try:
        namespace = next((namespace for namespace in DATA['metadef_namespaces'] if namespace['namespace'] == namespace_name))
    except StopIteration:
        LOG.debug('No namespace found with name %s', namespace_name)
        raise exception.MetadefNamespaceNotFound(namespace_name=namespace_name)
    _check_namespace_visibility(context, namespace, namespace_name)
    return namespace