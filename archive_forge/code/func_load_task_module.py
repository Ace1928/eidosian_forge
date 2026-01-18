from typing import Callable, Dict, Type
import importlib
from collections import namedtuple
def load_task_module(taskname: str):
    """
    Get the module containing all teacher agents for the task specified by `--task`.

    :param taskname: path to task class in one of the following formats:
        * full: ``-t parlai.tasks.babi.agents:DefaultTeacher``
        * shorthand: ``-t babi``, which will check
            ``parlai.tasks.babi.agents:DefaultTeacher``
        * shorthand specific: ``-t babi:task10k``, which will check
            ``parlai.tasks.babi.agents:Task10kTeacher``

    :return:
        module containing all teacher agents for a task
    """
    task_path_list, repo = _get_task_path_and_repo(taskname)
    task_path = task_path_list[0]
    if '.' in task_path:
        module_name = task_path
    else:
        task = task_path.lower()
        module_name = '%s.tasks.%s.agents' % (repo, task)
    task_module = importlib.import_module(module_name)
    return task_module