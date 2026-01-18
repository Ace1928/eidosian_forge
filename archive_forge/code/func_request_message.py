import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def request_message(self):
    if not (self.disconnected or self.some_agent_disconnected or self.hit_is_expired):
        self.m_send_command(self.worker_id, self.assignment_id, {'text': data_model.COMMAND_SEND_MESSAGE})