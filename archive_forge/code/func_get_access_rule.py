import abc
from keystone import exception
@abc.abstractmethod
def get_access_rule(self, access_rule_id):
    """Get an access rule by its ID.

        :param str access_rule_id: Access Rule ID
        """
    raise exception.NotImplemented()