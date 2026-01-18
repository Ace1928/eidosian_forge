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
def requeue_task_data(self, worker_id: str, task_data: List[Dict[str, Any]]):
    """
        Return task to task_queue.

        If the task is an onboarding task, indicate that the worker has
        another onboarding task to do.

        :param worker_id:
            worker id of worker who is returning task

        :param task_data:
            list of unfinished tasks to return to the queue.
        """
    worker_data = self._get_worker_data(worker_id)
    for subtask_data in task_data:
        if subtask_data['task_specs'].get('is_onboarding', False):
            worker_data['onboarding_todo'].append(subtask_data['pair_id'])
        else:
            self.task_queue.put(subtask_data)
            try:
                worker_data['tasks_completed'].remove(subtask_data['pair_id'])
                for d_id in self._get_dialogue_ids(subtask_data):
                    worker_data['conversations_seen'].remove(d_id)
            except ValueError:
                warn_once(f'could not remove task from worker {worker_id} history')