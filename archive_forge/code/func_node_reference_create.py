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
def node_reference_create(context, node_reference_url, **values):
    global DATA
    node_reference_id = values.get('node_reference_id', 1)
    if node_reference_id in DATA['node_reference']:
        raise exception.Duplicate()
    node_reference = {'node_reference_id': node_reference_id, 'node_reference_url': node_reference_url}
    DATA['node_reference'][node_reference_id] = node_reference
    return node_reference