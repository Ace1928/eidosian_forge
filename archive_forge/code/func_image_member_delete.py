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
def image_member_delete(context, member_id):
    global DATA
    for i, member in enumerate(DATA['members']):
        if member['id'] == member_id:
            del DATA['members'][i]
            break
    else:
        raise exception.NotFound()