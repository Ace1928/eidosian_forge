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
def reject_work(self, assignment_id, reason):
    """
        reject work for a given assignment through the mturk client.
        """
    client = mturk_utils.get_mturk_client(self.is_sandbox)
    client.reject_assignment(AssignmentId=assignment_id, RequesterFeedback=reason)
    if self.db_logger is not None:
        self.db_logger.log_reject_assignment(assignment_id)
    shared_utils.print_and_log(logging.INFO, 'Assignment {} rejected for reason {}.'.format(assignment_id, reason))