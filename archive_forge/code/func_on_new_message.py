import logging
import os
import threading
import time
import uuid
from parlai.mturk.core.shared_utils import AssignState
from parlai.mturk.core.socket_manager import Packet
from parlai.mturk.webapp.run_mocks.mock_turk_agent import MockTurkAgent
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def on_new_message(self, worker_id, msg):
    agent = self.id_to_agent[worker_id]
    agent.put_data(msg.id, msg.data)
    agent.append_message(msg.data)