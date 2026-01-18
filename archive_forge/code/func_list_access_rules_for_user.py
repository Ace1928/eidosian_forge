import abc
from keystone import exception
@abc.abstractmethod
def list_access_rules_for_user(self, user_id):
    """List the access rules that a user has created.

        Access rules are only created as attributes of application credentials,
        they cannot be created independently.

        :param str user_id: User ID
        """
    raise exception.NotImplemented()