from openstack.message.v2 import claim as _claim
from openstack.message.v2 import message as _message
from openstack.message.v2 import queue as _queue
from openstack.message.v2 import subscription as _subscription
from openstack import proxy
from openstack import resource
def get_claim(self, queue_name, claim):
    """Get a claim

        :param queue_name: The name of target queue to claim message from.
        :param claim: The value can be either the ID of a claim or a
            :class:`~openstack.message.v2.claim.Claim` instance.

        :returns: One :class:`~openstack.message.v2.claim.Claim`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            claim matching the criteria could be found.
        """
    return self._get(_claim.Claim, claim, queue_name=queue_name)