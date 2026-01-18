import abc
from keystone import exception
def delete_federated_object(self, user_id):
    """Delete a user's federated objects.

        :param user_id: Unique identifier of the user
        """
    raise exception.NotImplemented()