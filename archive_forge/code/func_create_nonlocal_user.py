import abc
from keystone import exception
@abc.abstractmethod
def create_nonlocal_user(self, user_dict):
    """Create a new non-local user.

        :param dict user_dict: Reference to the non-local user
        :returns dict: Containing the user reference

        """
    raise exception.NotImplemented()