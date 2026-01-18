import abc
from keystone import exception
@abc.abstractmethod
def create_federated_object(self, fed_dict):
    """Create a new federated object.

        :param dict federated_dict: Reference to the federated user
        """
    raise exception.NotImplemented()