from oslo_utils import uuidutils
from zaqarclient.common import decorators
from zaqarclient.queues.v1 import core
from zaqarclient.queues.v1 import flavor
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import pool
from zaqarclient.queues.v1 import queues
from zaqarclient import transport
from zaqarclient.transport import errors
from zaqarclient.transport import request
def pools(self, **params):
    """Gets a list of pools from the server

        :param params: Filters to use for getting pools
        :type params: dict.

        :returns: A list of pools
        :rtype: `list`
        """
    req, trans = self._request_and_transport()
    pool_list = core.pool_list(trans, req, **params)
    return iterator._Iterator(self, pool_list, 'pools', pool.create_object(self))