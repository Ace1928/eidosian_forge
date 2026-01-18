import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def get_last_command(self):
    """
        Returns the last command to be sent to this agent.
        """
    return self.state.get_last_command()