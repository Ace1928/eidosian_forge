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
def close_all_channels(self):
    """
        Closes all channels by clearing the list of channels.
        """
    shared_utils.print_and_log(logging.DEBUG, 'Closing all channels')
    connection_ids = list(self.open_channels)
    for connection_id in connection_ids:
        self.close_channel(connection_id)