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
def metadef_tag_delete(context, namespace_name, name):
    """Delete a metadef tag"""
    global DATA
    tags = metadef_tag_get(context, namespace_name, name)
    DATA['metadef_tags'].remove(tags)
    return tags