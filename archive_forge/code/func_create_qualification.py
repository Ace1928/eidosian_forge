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
def create_qualification(self, qualification_name, description, can_exist=True):
    """
        Create a new qualification.

        If can_exist is set, simply return the ID of the existing qualification rather
        than throw an error
        """
    if not can_exist:
        qual_id = mturk_utils.find_qualification(qualification_name, self.is_sandbox)
        if qual_id is not None:
            shared_utils.print_and_log(logging.WARN, 'Could not create qualification {}, as it existed'.format(qualification_name), should_print=True)
            return None
    return mturk_utils.find_or_create_qualification(qualification_name, description, self.is_sandbox)