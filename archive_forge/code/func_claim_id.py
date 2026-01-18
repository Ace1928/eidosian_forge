from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import core
@property
def claim_id(self):
    if '=' in self.href:
        return self.href.split('=')[-1]