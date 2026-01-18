import abc
from keystone import exception
@abc.abstractmethod
def list_associations_for_policy(self, policy_id):
    """List the associations for a policy.

        This method is not exposed as a public API, but is used by
        list_endpoints_for_policy().

        :param policy_id: identity of policy
        :type policy_id: string
        :returns: List of association dicts

        """
    raise exception.NotImplemented()