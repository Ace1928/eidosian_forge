import re
from kombu.utils.encoding import safe_str
def task_args_contains_search_args(task_args, search_args):
    if not task_args:
        return False
    return all((a in task_args for a in search_args))