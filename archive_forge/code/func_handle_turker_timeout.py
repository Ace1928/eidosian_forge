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
def handle_turker_timeout(self, worker_id, assign_id):
    """
        To be used by the MTurk agent when the worker doesn't send a message within the
        expected window.
        """
    text = "You haven't entered a message in too long. As these HITs  often require real-time interaction, this hit has been expired and you have been considered disconnected. Disconnect too frequently and you will be blocked from working on these HITs in the future."
    self.force_expire_hit(worker_id, assign_id, text)
    self._handle_agent_disconnect(worker_id, assign_id)