import abc
from keystone import exception
@abc.abstractmethod
def create_sp(self, sp_id, sp):
    """Create a service provider.

        :param sp_id: id of the service provider
        :type sp_id: string
        :param sp: service provider object
        :type sp: dict

        :returns: service provider ref
        :rtype: dict

        """
    raise exception.NotImplemented()