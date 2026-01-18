import abc
from keystone import exception
@abc.abstractmethod
def delete_association_by_policy(self, policy_id):
    """Remove all the policy associations with the specific policy.

        :param policy_id: identity of endpoint to check
        :type policy_id: string
        :returns: None

        """
    raise exception.NotImplemented()