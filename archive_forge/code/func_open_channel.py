import errno
import logging
import json
import threading
import time
from queue import PriorityQueue, Empty
import websocket
from parlai.mturk.core.dev.shared_utils import print_and_log
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def open_channel(self, worker_id, assignment_id):
    """
        Opens a channel for a worker on a given assignment, doesn't re-open if the
        channel is already open.
        """
    connection_id = '{}_{}'.format(worker_id, assignment_id)
    self.open_channels.add(connection_id)
    self.worker_assign_ids[connection_id] = (worker_id, assignment_id)