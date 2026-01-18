import abc
from keystone import exception
@abc.abstractmethod
def delete_access_rule(self, access_rule_id):
    """Delete one access rule.

        :param str access_rule_id: Access Rule ID
        """
    raise exception.NotImplemented()