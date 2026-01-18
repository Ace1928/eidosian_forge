import errno
import logging
import json
import threading
import time
from queue import PriorityQueue, Empty
import websocket
from parlai.mturk.core.shared_utils import print_and_log
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def packet_should_block(self, packet_item):
    """
        Helper function to determine if a packet is still blocking.
        """
    t, packet = packet_item
    if time.time() > t:
        return False
    if packet.status in [Packet.STATUS_ACK, Packet.STATUS_FAIL]:
        return False
    return True