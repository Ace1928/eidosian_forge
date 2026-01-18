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
def create_hits(self, qualifications=None):
    """
        Create hits based on the managers current config, return hit url.
        """
    shared_utils.print_and_log(logging.INFO, 'Creating HITs...', True)
    if self.task_state < self.STATE_ACCEPTING_WORKERS:
        shared_utils.print_and_log(logging.WARN, 'You should be calling `ready_to_accept_workers` before `create_hits` to ensure that the socket is connected beforehits are added. This will be enforced in future versions.', True)
    if self.opt['max_connections'] == 0:
        mturk_page_url = self.create_additional_hits(num_hits=self.required_hits, qualifications=qualifications)
    else:
        mturk_page_url = self.create_additional_hits(num_hits=min(self.required_hits, self.opt['max_connections']), qualifications=qualifications)
    shared_utils.print_and_log(logging.INFO, 'Link to HIT: {}\n'.format(mturk_page_url), should_print=True)
    shared_utils.print_and_log(logging.INFO, "Waiting for Turkers to respond... (Please don't close your laptop or put your computer into sleep or standby mode.)\n", should_print=True)
    self.task_state = self.STATE_HITS_MADE
    return mturk_page_url