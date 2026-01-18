import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def get_inactive_command_data(self):
    """
        Get appropriate inactive command data to respond to a reconnect.
        """
    text, command = self.state.get_inactive_command_text()
    return {'text': command, 'inactive_text': text, 'conversation_id': self.conversation_id, 'agent_id': self.worker_id}