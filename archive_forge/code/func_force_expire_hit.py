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
def force_expire_hit(self, worker_id, assign_id, text=None, ack_func=None):
    """
        Send a command to expire a hit to the provided agent, update State to reflect
        that the HIT is now expired.
        """
    agent = self.worker_manager._get_agent(worker_id, assign_id)
    if agent is not None:
        if agent.is_final():
            return
        agent.set_status(AssignState.STATUS_EXPIRED)
        agent.hit_is_expired = True
    if ack_func is None:

        def use_ack_func(*args):
            self.socket_manager.close_channel('{}_{}'.format(worker_id, assign_id))
    else:

        def use_ack_func(*args):
            ack_func(*args)
            self.socket_manager.close_channel('{}_{}'.format(worker_id, assign_id))
    if text is None:
        text = "This HIT is expired, please return and take a new one if you'd want to work on this task."
    data = {'agent_status': AssignState.STATUS_EXPIRED, 'done_text': text}
    self.send_state_change(worker_id, assign_id, data, ack_func=use_ack_func)