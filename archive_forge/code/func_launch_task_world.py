import logging
import time
import datetime
from concurrent import futures
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils
def launch_task_world(self, task_name, world_name, agents):
    """
        Launch a task world.

        Return the job's future.

        :param task_name:
            string. the name of the job thread
        :param world_name:
            string. the name of the task world in the module file
        :param agents:
            list. the list of agents to install in the world

        :return:
            the Futures object corresponding to this launched task
        """
    task = utils.TaskState(task_name, world_name, agents)
    self.tasks[task_name] = task

    def _world_fn():
        log_utils.print_and_log(logging.INFO, 'Starting task {}...'.format(task_name))
        return self._run_world(task, world_name, agents)
    fut = self.executor.submit(_world_fn)
    task.future = fut
    return fut