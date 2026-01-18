from openstack.message.v2 import claim as _claim
from openstack.message.v2 import message as _message
from openstack.message.v2 import queue as _queue
from openstack.message.v2 import subscription as _subscription
from openstack import proxy
from openstack import resource
def create_claim(self, queue_name, **attrs):
    """Create a new claim from attributes

        :param queue_name: The name of target queue to claim message from.
        :param dict attrs: Keyword arguments which will be used to create a
            :class:`~openstack.message.v2.claim.Claim`,
            comprised of the properties on the Claim class.

        :returns: The results of claim creation
        :rtype: :class:`~openstack.message.v2.claim.Claim`
        """
    return self._create(_claim.Claim, queue_name=queue_name, **attrs)