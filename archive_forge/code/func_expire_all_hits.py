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
def expire_all_hits(self):
    """
        Move through all the HITs launched and expire every one of them.

        Ensures no HITs should be around after this instance is dead
        """
    shared_utils.print_and_log(logging.INFO, 'Expiring all HITs...', should_print=not self.is_test)
    for hit_id in self.all_hit_ids:
        mturk_utils.expire_hit(self.is_sandbox, hit_id)