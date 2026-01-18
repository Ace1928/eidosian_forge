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
def check_and_update_worker_approval(self, worker_id: str, save_data: Dict[str, Any]):
    """
        Soft block workers who fail onboarding tasks, keep track of their status.

        :param worker_id:
            worker id

        :param save_data:
            data from the worker's completed tasks
        """
    all_task_data = save_data['worker_data'][worker_id]['task_data']
    response_data = save_data['worker_data'][worker_id]['response']['task_data']
    num_onboarding_tasks = 0
    num_correct = 0
    for i in range(len(all_task_data)):
        is_onboarding = all_task_data[i]['pairing_dict'].get('is_onboarding', False)
        if not is_onboarding:
            continue
        worker_response = response_data[i]['speakerChoice']
        expected_response = all_task_data[i]['pairing_dict']['correct_answer']
        num_onboarding_tasks += 1
        if worker_response == expected_response:
            num_correct += 1
    if num_onboarding_tasks == 0:
        if worker_id in self.failed_onboard:
            self.requeue_task_data(worker_id, all_task_data)
        return
    if num_correct / num_onboarding_tasks >= self.opt['onboarding_threshold']:
        return
    self.manager.soft_block_worker(worker_id)
    self.failed_onboard.add(worker_id)