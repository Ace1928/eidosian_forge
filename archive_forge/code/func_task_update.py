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
def task_update(context, task_id, values):
    """Update a task object"""
    global DATA
    task_values = copy.deepcopy(values)
    task_info_values = _pop_task_info_values(task_values)
    try:
        task = DATA['tasks'][task_id]
    except KeyError:
        LOG.debug('No task found with ID %s', task_id)
        raise exception.TaskNotFound(task_id=task_id)
    task.update(task_values)
    task['updated_at'] = timeutils.utcnow()
    DATA['tasks'][task_id] = task
    task_info = _task_info_update(task['id'], task_info_values)
    return _format_task_from_db(task, task_info)