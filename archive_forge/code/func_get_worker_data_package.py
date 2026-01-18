import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def get_worker_data_package(self):
    workers = []
    for mturk_worker in self.mturk_workers.values():
        worker = {'id': mturk_worker.worker_id, 'accepted': len(mturk_worker.agents), 'completed': mturk_worker.completed_assignments(), 'disconnected': mturk_worker.disconnected_assignments()}
        workers.append(worker)
    return workers