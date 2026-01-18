from oslo_utils import uuidutils
from zaqarclient.common import decorators
from zaqarclient.queues.v1 import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v2 import core
from zaqarclient.queues.v2 import queues
from zaqarclient.queues.v2 import subscription
@decorators.version(min_version=1.1)
def homedoc(self):
    """Get the detailed resource doc of Zaqar server"""
    req, trans = self._request_and_transport()
    return core.homedoc(trans, req)