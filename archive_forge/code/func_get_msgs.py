import sys
from queue import Empty, Queue
from traitlets import Type
from .channels import InProcessChannel
from .client import InProcessKernelClient
def get_msgs(self):
    """Get all messages that are currently ready."""
    msgs = []
    while True:
        try:
            msgs.append(self.get_msg(block=False))
        except Empty:
            break
    return msgs