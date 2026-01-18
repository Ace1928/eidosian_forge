import abc
from keystone import exception
@abc.abstractmethod
def delete_sp(self, sp_id):
    """Delete a service provider.

        :param sp_id: id of the service provider
        :type sp_id: string

        :raises keystone.exception.ServiceProviderNotFound: If the service
            provider doesn't exist.

        """
    raise exception.NotImplemented()