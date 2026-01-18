from oslo_utils import uuidutils
from zaqarclient.common import decorators
from zaqarclient.queues.v1 import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v2 import core
from zaqarclient.queues.v2 import queues
from zaqarclient.queues.v2 import subscription
@decorators.version(min_version=2)
def subscription(self, queue_name, **kwargs):
    """Returns a subscription instance

        :param queue_name: Name of the queue to subscribe to.
        :type queue_name: str

        :returns: A subscription instance
        :rtype: `subscription.Subscription`
        """
    return subscription.Subscription(self, queue_name, **kwargs)