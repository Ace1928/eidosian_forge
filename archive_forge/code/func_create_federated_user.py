import abc
from keystone import exception
@abc.abstractmethod
def create_federated_user(self, domain_id, federated_dict, email=None):
    """Create a new user with the federated identity.

        :param domain_id: The domain ID of the IdP used for the federated user
        :param dict federated_dict: Reference to the federated user
        :param email: Federated user's email
        :returns dict: Containing the user reference

        """
    raise exception.NotImplemented()