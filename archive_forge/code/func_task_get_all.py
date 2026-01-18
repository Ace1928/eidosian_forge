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
def task_get_all(context, filters=None, marker=None, limit=None, sort_key='created_at', sort_dir='desc'):
    """
    Get all tasks that match zero or more filters.

    :param filters: dict of filter keys and values.
    :param marker: task id after which to start page
    :param limit: maximum number of tasks to return
    :param sort_key: task attribute by which results should be sorted
    :param sort_dir: direction in which results should be sorted (asc, desc)
    :returns: tasks set
    """
    _task_soft_delete(context)
    filters = filters or {}
    tasks = DATA['tasks'].values()
    tasks = _filter_tasks(tasks, filters, context)
    tasks = _sort_tasks(tasks, sort_key, sort_dir)
    tasks = _paginate_tasks(context, tasks, marker, limit, filters.get('deleted'))
    filtered_tasks = []
    for task in tasks:
        filtered_tasks.append(_format_task_from_db(task, task_info_ref=None))
    return filtered_tasks