import threading
from .. import cethread, tests
def set_sync_event(self, event):
    if event is self.step2:
        raise MyException()
    super().set_sync_event(event)