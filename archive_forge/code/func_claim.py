from zaqarclient.queues.v1 import queues
from zaqarclient.queues.v2 import claim as claim_api
from zaqarclient.queues.v2 import core
from zaqarclient.queues.v2 import message
def claim(self, id=None, ttl=None, grace=None, limit=None):
    return claim_api.Claim(self, id=id, ttl=ttl, grace=grace, limit=limit)