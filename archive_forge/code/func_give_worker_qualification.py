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
def give_worker_qualification(self, worker_id, qual_name, qual_value=None):
    """
        Give a worker a particular qualification.
        """
    qual_id = mturk_utils.find_qualification(qual_name, self.is_sandbox)
    if qual_id is False or qual_id is None:
        shared_utils.print_and_log(logging.WARN, 'Could not give worker {} qualification {}, as the qualification could not be found to exist.'.format(worker_id, qual_name), should_print=True)
        return
    mturk_utils.give_worker_qualification(worker_id, qual_id, qual_value, self.is_sandbox)
    shared_utils.print_and_log(logging.INFO, 'gave {} qualification {}'.format(worker_id, qual_name), should_print=True)