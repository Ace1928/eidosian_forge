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
@staticmethod
def make_taskless_instance(is_sandbox=False):
    """
        Creates an instance without a task to be used for approving or rejecting
        assignments, blocking workers, and managing qualifications.
        """
    opt = {'unique_worker': False, 'max_hits_per_worker': 0, 'num_conversations': 0, 'is_sandbox': is_sandbox, 'is_debug': False, 'log_level': 30}
    manager = MTurkManager(opt, [], use_db=True)
    manager.is_shutdown.set()
    mturk_utils.setup_aws_credentials()
    return manager