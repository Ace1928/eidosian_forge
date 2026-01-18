import queue
from threading import RLock
from ..parser import Parser
def iterpoll(self):
    while True:
        msg = self.poll()
        if msg is None:
            return
        else:
            yield msg