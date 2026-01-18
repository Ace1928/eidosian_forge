import abc
from keystone import exception
@abc.abstractmethod
def get_federated_user(self, idp_id, protocol_id, unique_id):
    """Return the found user for the federated identity.

        :param idp_id: The identity provider ID
        :param protocol_id: The federation protocol ID
        :param unique_id: The unique ID for the user
        :returns dict: Containing the user reference

        """
    raise exception.NotImplemented()