import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def check_system_grant(self, role_id, actor_id, target_id, inherited):
    """Check if a user or group has a specific role on the system.

        :param role_id: the unique ID of the role to grant to the user
        :param actor_id: the unique ID of the user or group
        :param target_id: the unique ID or string representing the target
        :param inherited: a boolean denoting if the assignment is inherited or
                          not

        """
    raise exception.NotImplemented()