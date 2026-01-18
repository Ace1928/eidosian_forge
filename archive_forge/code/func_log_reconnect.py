import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def log_reconnect(self):
    """
        Log a reconnect of this agent.
        """
    shared_utils.print_and_log(logging.DEBUG, 'Agent ({})_({}) reconnected to {} with status {}'.format(self.worker_id, self.assignment_id, self.conversation_id, self.get_status()))