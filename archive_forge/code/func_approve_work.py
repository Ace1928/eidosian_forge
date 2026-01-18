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
def approve_work(self, assignment_id, override_rejection=False):
    """
        approve work for a given assignment through the mturk client.
        """
    client = mturk_utils.get_mturk_client(self.is_sandbox)
    client.approve_assignment(AssignmentId=assignment_id, OverrideRejection=override_rejection)
    if self.db_logger is not None:
        self.db_logger.log_approve_assignment(assignment_id)
    shared_utils.print_and_log(logging.INFO, 'Assignment {} approved.'.format(assignment_id))