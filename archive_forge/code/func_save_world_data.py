import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
@staticmethod
def save_world_data(prepped_save_data, task_group_id, conversation_id, sandbox=False):
    target = 'sandbox' if sandbox else 'live'
    if task_group_id is None:
        return
    target_dir = os.path.join(data_dir, target, task_group_id, conversation_id)
    custom_data = prepped_save_data['custom_data']
    if custom_data is not None:
        target_dir_custom = os.path.join(target_dir, 'custom')
        if custom_data.get('needs-pickle') is not None:
            pickle_file = os.path.join(target_dir_custom, 'data.pickle')
            force_dir(pickle_file)
            with open(pickle_file, 'wb') as outfile:
                pickle.dump(custom_data['needs-pickle'], outfile)
            del custom_data['needs-pickle']
        custom_file = os.path.join(target_dir_custom, 'data.json')
        force_dir(custom_file)
        print('Saving data to {}.'.format(custom_file))
        with open(custom_file, 'w') as outfile:
            json.dump(custom_data, outfile)
    worker_data = prepped_save_data['worker_data']
    target_dir_workers = os.path.join(target_dir, 'workers')
    for worker_id, w_data in worker_data.items():
        worker_file = os.path.join(target_dir_workers, '{}.json'.format(worker_id))
        force_dir(worker_file)
        with open(worker_file, 'w') as outfile:
            json.dump(w_data, outfile)