from typing import Callable, Dict, Type
import importlib
from collections import namedtuple
def load_teacher_module(taskname: str):
    """
    Get the module of the teacher agent specified by `--task`.

    Can be formatted in several different ways:

    * full: ``-t parlai.tasks.babi.agents:DefaultTeacher``
    * shorthand: ``-t babi``, which will check
        ``parlai.tasks.babi.agents:DefaultTeacher``
    * shorthand specific: ``-t babi:task10k``, which will check
        ``parlai.tasks.babi.agents:Task10kTeacher``

    The base path to search when using shorthand formats can be changed from
    "parlai" to "parlai_internal" by prepending "internal:" to the path, e.g.
    "internal:babi".

    Options can be sent to the teacher by adding an additional colon,
    for example ``-t babi:task10k:1`` directs the babi Task10kTeacher to use
    task number 1.

    :param taskname: path to task class in one of the above formats.

    :return:
        teacher module
    """
    global TEACHER_REGISTRY
    if taskname in TEACHER_REGISTRY:
        return TEACHER_REGISTRY[taskname]
    task_module = load_task_module(taskname)
    task_path_list, repo = _get_task_path_and_repo(taskname)
    if len(task_path_list) > 1 and '=' not in task_path_list[1]:
        task_path_list[1] = task_path_list[1][0].upper() + task_path_list[1][1:]
        teacher = task_path_list[1]
        if '.' not in task_path_list[0] and 'Teacher' not in teacher:
            words = teacher.split('_')
            teacher_name = ''
            for w in words:
                teacher_name += w[0].upper() + w[1:]
            teacher = teacher_name + 'Teacher'
    else:
        teacher = 'DefaultTeacher'
    teacher_class = getattr(task_module, teacher)
    return teacher_class