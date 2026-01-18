import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def submitted_hit(self):
    return self.get_status() in [AssignState.STATUS_DONE, AssignState.STATUS_PARTNER_DISCONNECT]