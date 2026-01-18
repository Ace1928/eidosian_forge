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
def email_worker(self, worker_id, subject, message_text):
    """
        Send an email to a worker through the mturk client.
        """
    client = mturk_utils.get_mturk_client(self.is_sandbox)
    response = client.notify_workers(Subject=subject, MessageText=message_text, WorkerIds=[worker_id])
    if len(response['NotifyWorkersFailureStatuses']) > 0:
        failure_message = response['NotifyWorkersFailureStatuses'][0]
        return {'failure': failure_message['NotifyWorkersFailureMessage']}
    else:
        return {'success': True}