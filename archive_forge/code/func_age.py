from zaqarclient.queues.v1 import core
from zaqarclient.queues.v1 import iterator as iterate
from zaqarclient.queues.v1 import message
@property
def age(self):
    self._get()
    return self._age