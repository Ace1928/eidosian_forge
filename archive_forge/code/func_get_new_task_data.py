from typing import Dict, List, Set, Any
import json
import os
import queue
import random
import time
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import StaticMTurkManager
from parlai.mturk.core.worlds import StaticMTurkTaskWorld
from parlai.utils.misc import warn_once
def get_new_task_data(self, worker_id: str) -> List[Dict[str, Any]]:
    """
        Get next task for worker.

        Returns the next onboarding task if worker hasn't finished them all,
        Otherwise finds a task from the queue they haven't seen

        If they've seen everything in the queue, spin up an
        extra task (one that was in the queue and is now saturated)

        :param worker_id:
            worker id

        :return task_data:
            A list of tasks for the worker to complete
        """
    tasks_per_hit = self.opt['subtasks_per_hit']
    task_data = self.get_onboarding_tasks(worker_id)
    if len(task_data) == tasks_per_hit:
        return task_data
    task_data = self._poll_task_queue(worker_id, task_data)
    if len(task_data) == tasks_per_hit:
        return task_data
    task_data = self._top_up_task_data(worker_id, task_data)
    return task_data