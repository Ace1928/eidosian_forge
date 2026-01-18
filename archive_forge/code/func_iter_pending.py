import random
import threading
import time
from .messages import Message
from .parser import Parser
def iter_pending(self):
    """Iterate through pending messages."""
    while True:
        msg = self.poll()
        if msg is None:
            return
        else:
            yield msg