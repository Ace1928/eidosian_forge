from zaqarclient.queues.v1 import queues
from zaqarclient.queues.v2 import claim as claim_api
from zaqarclient.queues.v2 import core
from zaqarclient.queues.v2 import message
def signed_url(self, paths=None, ttl_seconds=None, methods=None):
    req, trans = self.client._request_and_transport()
    return core.signed_url_create(trans, req, self._name, paths=paths, ttl_seconds=ttl_seconds, methods=methods)