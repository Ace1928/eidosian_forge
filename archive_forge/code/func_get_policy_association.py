import abc
from keystone import exception
@abc.abstractmethod
def get_policy_association(self, endpoint_id=None, service_id=None, region_id=None):
    """Get the policy for an explicit association.

        This method is not exposed as a public API, but is used by
        get_policy_for_endpoint().

        :param endpoint_id: identity of endpoint
        :type endpoint_id: string
        :param service_id: identity of the service
        :type service_id: string
        :param region_id: identity of the region
        :type region_id: string
        :raises keystone.exception.PolicyAssociationNotFound: If there is no
            match for the specified association.
        :returns: dict containing policy_id (value is a tuple containing only
                  the policy_id)

        """
    raise exception.NotImplemented()