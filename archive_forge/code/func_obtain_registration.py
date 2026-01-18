import abc
from keystone import exception
@abc.abstractmethod
def obtain_registration(self, domain_id, type):
    """Try and register this domain to use the type specified.

        :param domain_id: the domain required
        :param type: type of registration
        :returns: True if the domain was registered, False otherwise. Failing
                  to register means that someone already has it (which could
                  even be the domain being requested).

        """
    raise exception.NotImplemented()