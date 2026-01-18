from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.utils.misc import AttrDict
from parlai.mturk.core.mturk_manager import MTurkManager
import parlai.mturk.core.mturk_utils as mturk_utils
from worlds import WizardEval, TopicsGenerator, TopicChooseWorld
from task_config import task_config
from projects.wizard_of_wikipedia.knowledge_retriever.knowledge_retriever import (
import gc
import datetime
import json
import os
import sys
from parlai.utils.logging import ParlaiLogger, INFO
def check_multiple_workers_eligibility(workers):
    valid_workers = {}
    for worker in workers:
        worker_id = worker.worker_id
        if worker_id not in worker_roles:
            print('Something went wrong')
            continue
        role = worker_roles[worker_id]
        if role not in valid_workers:
            valid_workers[role] = worker
        if len(valid_workers) == 2:
            break
    return valid_workers.values() if len(valid_workers) == 2 else []