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
def start_new_run(self):
    """
        Clear state to prepare for a new run.
        """
    assert self.task_state >= self.STATE_SERVER_ALIVE, 'Cannot start a run before having a running server using `mturk_manager.setup_server()` first.'
    self.run_id = str(int(time.time()))
    self.task_group_id = '{}_{}'.format(self.opt['task'], self.run_id)
    self._init_state()
    try:
        self.topic_arn = mturk_utils.setup_sns_topic(self.opt['task'], self.server_url, self.task_group_id)
    except Exception as e:
        self.topic_arn = None
        shared_utils.print_and_log(logging.WARN, "Botocore couldn't subscribe to HIT events, perhaps you tried to register to localhost?", should_print=True)
        print(repr(e))
    if self.db_logger is not None:
        self.db_logger.log_new_run(self.required_hits, self.opt['task'])
    self.task_state = self.STATE_INIT_RUN