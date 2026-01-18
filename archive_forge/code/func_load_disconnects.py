import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def load_disconnects(self):
    """
        Load disconnects from file, populate the disconnects field for any worker_id
        that has disconnects in the list.

        Any disconnect that occurred longer ago than the disconnect persist length is
        ignored
        """
    self.disconnects = []
    file_path = os.path.join(parent_dir, DISCONNECT_FILE_NAME)
    compare_time = time.time()
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            old_disconnects = pickle.load(f)
            self.disconnects = [d for d in old_disconnects if compare_time - d['time'] < DISCONNECT_PERSIST_LENGTH]
    for disconnect in self.disconnects:
        worker_id = disconnect['id']
        if worker_id not in self.mturk_workers:
            self.mturk_workers[worker_id] = WorkerState(worker_id)
        self.mturk_workers[worker_id].disconnects += 1