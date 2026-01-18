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
def channel_thread(self):
    """
        Handler thread for monitoring all channels.
        """
    while not self.is_shutdown:
        if self.ws is None:
            time.sleep(shared_utils.THREAD_SHORT_SLEEP)
            continue
        self._try_send_world_ping()
        try:
            item = self.sending_queue.get(block=False)
            t = item[0]
            if time.time() < t:
                self.sending_queue.put(item)
            else:
                packet = item[1]
                if not packet:
                    continue
                if packet.status is not Packet.STATUS_SENT:
                    self._send_packet(packet, t)
        except Empty:
            time.sleep(shared_utils.THREAD_SHORT_SLEEP)
        except Exception as e:
            shared_utils.print_and_log(logging.WARN, 'Unexpected error occurred in socket handling thread: {}'.format(repr(e)), should_print=True)