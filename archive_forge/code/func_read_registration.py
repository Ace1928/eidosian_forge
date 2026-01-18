import abc
from keystone import exception
@abc.abstractmethod
def read_registration(self, type):
    """Get the domain ID of who is registered to use this type.

        :param type: type of registration
        :returns: domain_id of who is registered.
        :raises keystone.exception.ConfigRegistrationNotFound: If nobody is
            registered.

        """
    raise exception.NotImplemented()