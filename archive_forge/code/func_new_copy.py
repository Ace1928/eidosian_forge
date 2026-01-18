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
def new_copy(self):
    """
        Return a new packet that is a copy of this packet with a new id and with a fresh
        status.
        """
    packet = Packet.from_dict(self.as_dict())
    packet.id = shared_utils.generate_event_id(self.receiver_id)
    return packet