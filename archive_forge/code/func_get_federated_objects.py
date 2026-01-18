import abc
from keystone import exception
@abc.abstractmethod
def get_federated_objects(self, user_id):
    """Get all federated objects for a user.

        :param user_id: Unique identifier of the user
        :returns list: Containing the user's federated objects

        """
    raise exception.NotImplemented()