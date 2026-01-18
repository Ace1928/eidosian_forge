import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def prepare_timeout(self):
    """
        Log a timeout event, tell mturk manager it occurred, return message to return
        for the act call.
        """
    shared_utils.print_and_log(logging.INFO, '{} timed out before sending.'.format(self.id))
    self.mturk_manager.handle_turker_timeout(self.worker_id, self.assignment_id)
    return self._get_episode_done_msg(TIMEOUT_MESSAGE)