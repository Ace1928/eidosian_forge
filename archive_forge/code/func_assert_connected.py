import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def assert_connected(self):
    """
        Ensures that an agent is still connected.
        """
    if self.disconnected or self.some_agent_disconnected:
        raise AgentDisconnectedError(self.worker_id, self.assignment_id)
    if self.hit_is_returned:
        raise AgentReturnedError(self.worker_id, self.assignment_id)
    return