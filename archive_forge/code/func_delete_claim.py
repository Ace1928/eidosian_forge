from openstack.message.v2 import claim as _claim
from openstack.message.v2 import message as _message
from openstack.message.v2 import queue as _queue
from openstack.message.v2 import subscription as _subscription
from openstack import proxy
from openstack import resource
def delete_claim(self, queue_name, claim, ignore_missing=True):
    """Delete a claim

        :param queue_name: The name of target queue to claim messages from.
        :param claim: The value can be either the ID of a claim or a
            :class:`~openstack.message.v2.claim.Claim` instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the claim does not exist.
            When set to ``True``, no exception will be thrown when
            attempting to delete a nonexistent claim.

        :returns: ``None``
        """
    return self._delete(_claim.Claim, claim, queue_name=queue_name, ignore_missing=ignore_missing)