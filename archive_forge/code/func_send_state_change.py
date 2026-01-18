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
def send_state_change(self, receiver_id, assignment_id, data, ack_func=None):
    """
        Send an updated state to the server to push to the agent.
        """
    event_id = shared_utils.generate_event_id(receiver_id)
    packet = Packet(event_id, data_model.AGENT_STATE_CHANGE, self.socket_manager.get_my_sender_id(), receiver_id, assignment_id, data, ack_func=ack_func)
    self.socket_manager.queue_packet(packet)