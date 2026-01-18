import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def set_last_command(self, command):
    """
        Changes the last command recorded as sent to the agent.
        """
    self.state.set_last_command(command)