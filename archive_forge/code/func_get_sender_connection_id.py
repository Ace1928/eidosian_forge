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
def get_sender_connection_id(self):
    """
        Get the connection_id that this packet came from.
        """
    return '{}_{}'.format(self.sender_id, self.assignment_id)