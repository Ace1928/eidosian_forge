import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.mturk_utils as mturk_utils
import parlai.mturk.core.dev.server_utils as server_utils
import parlai.mturk.core.dev.shared_utils as shared_utils
def mark_workers_done(self, workers):
    """
        Mark a group of agents as done to keep state consistent.
        """
    for agent in workers:
        if not agent.is_final():
            agent.set_status(AssignState.STATUS_DONE, 'done', None)
        if self.is_unique:
            assert self.unique_qual_name is not None, 'Unique qual name must not be none to use is_unique'
            self.give_worker_qualification(agent.worker_id, self.unique_qual_name)
        if self.max_hits_per_worker > 0:
            worker_state = self.worker_manager._get_worker(agent.worker_id)
            completed_assignments = worker_state.completed_assignments()
            assert self.unique_qual_name is not None, 'Unique qual name must not be none to use max_hits_per_worker'
            if completed_assignments >= self.max_hits_per_worker:
                self.give_worker_qualification(agent.worker_id, self.unique_qual_name)
        if self.has_time_limit:
            self._log_working_time(agent)