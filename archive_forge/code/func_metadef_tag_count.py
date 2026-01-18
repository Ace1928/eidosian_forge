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
def metadef_tag_count(context, namespace_name):
    """Get metadef tag count in a namespace"""
    namespace = metadef_namespace_get(context, namespace_name)
    _check_namespace_visibility(context, namespace, namespace_name)
    count = 0
    for tag in DATA['metadef_tags']:
        if tag['namespace_id'] == namespace['id']:
            count = count + 1
    return count