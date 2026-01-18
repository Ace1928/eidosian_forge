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
def image_member_update(context, member_id, values):
    global DATA
    for member in DATA['members']:
        if member['id'] == member_id:
            member.update(values)
            member['updated_at'] = timeutils.utcnow()
            return copy.deepcopy(member)
    else:
        raise exception.NotFound()