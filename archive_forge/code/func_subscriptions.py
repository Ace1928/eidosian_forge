from oslo_utils import uuidutils
from zaqarclient.common import decorators
from zaqarclient.queues.v1 import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v2 import core
from zaqarclient.queues.v2 import queues
from zaqarclient.queues.v2 import subscription
@decorators.version(min_version=2)
def subscriptions(self, queue_name, **params):
    """Gets a list of subscriptions from the server

        :param params: Filters to use for getting subscriptions
        :type params: dict.

        :returns: A list of subscriptions
        :rtype: `list`
        """
    req, trans = self._request_and_transport()
    subscription_list = core.subscription_list(trans, req, queue_name, **params)
    return iterator._Iterator(self, subscription_list, 'subscriptions', subscription.create_object(self))