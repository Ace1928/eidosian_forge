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
def remove_worker_qualification(self, worker_id, qual_name, reason=''):
    """
        Remove a qualification from a worker.
        """
    qual_id = mturk_utils.find_qualification(qual_name, self.is_sandbox)
    if qual_id is False or qual_id is None:
        shared_utils.print_and_log(logging.WARN, 'Could not remove from worker {} qualification {}, as the qualification could not be found to exist.'.format(worker_id, qual_name), should_print=True)
        return
    try:
        mturk_utils.remove_worker_qualification(worker_id, qual_id, self.is_sandbox, reason)
        shared_utils.print_and_log(logging.INFO, "removed {}'s qualification {}".format(worker_id, qual_name), should_print=True)
    except Exception as e:
        shared_utils.print_and_log(logging.WARN if not self.has_time_limit else logging.INFO, "removing {}'s qualification {} failed with error {}. This can be because the worker didn't have that qualification.".format(worker_id, qual_name, repr(e)), should_print=True)