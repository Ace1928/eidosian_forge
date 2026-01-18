from typing import Callable, Dict, Type
import importlib
from collections import namedtuple
def load_world_module(taskname: str, interactive_task: bool=False, selfchat_task: bool=False, num_agents: int=None, default_world=None):
    """
    Load the world module for the specific environment. If not enough information is to
    determine which world should be loaded, returns None.

    :param taskname:
        path to task class in one of the above formats
    :param interactive_task:
        whether or not the task is interactive
    :param num_agents:
        number of agents in the world; this may not be known a priori
    :param default_world:
        default world to return if specified

    :return:
        World module (or None, if not enough info to determine is present)
    """
    task = taskname.strip()
    repo = 'parlai'
    if task.startswith('internal:'):
        repo = 'parlai_internal'
        task = task[9:]
    task_path_list = task.split(':')
    if '.' in task_path_list[0]:
        return _get_default_world(default_world, num_agents)
    task = task_path_list[0].lower()
    if len(task_path_list) > 1:
        task_path_list[1] = task_path_list[1][0].upper() + task_path_list[1][1:]
        world_name = task_path_list[1] + 'World'
        if interactive_task:
            world_name = 'Interactive' + world_name
        elif selfchat_task:
            world_name = 'SelfChat' + world_name
    elif interactive_task:
        world_name = 'InteractiveWorld'
    elif selfchat_task:
        world_name = 'SelfChatWorld'
    else:
        world_name = 'DefaultWorld'
    module_name = '%s.tasks.%s.worlds' % (repo, task)
    try:
        my_module = importlib.import_module(module_name)
        world_class = getattr(my_module, world_name)
    except (ModuleNotFoundError, AttributeError):
        world_class = _get_default_world(default_world, num_agents)
    return world_class