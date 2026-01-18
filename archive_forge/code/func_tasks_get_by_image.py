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
def tasks_get_by_image(context, image_id):
    db_tasks = DATA['tasks']
    tasks = []
    for task in db_tasks:
        if db_tasks[task]['image_id'] == image_id:
            if _is_task_visible(context, db_tasks[task]):
                tasks.append(db_tasks[task])
    return tasks