import abc
from keystone import exception
@abc.abstractmethod
def list_sps(self, hints):
    """List all service providers.

        :param hints: filter hints which the driver should
                      implement if at all possible.
        :returns: List of service provider ref objects
        :rtype: list of dicts

        :raises keystone.exception.ServiceProviderNotFound: If the SP
            doesn't exist.

        """
    raise exception.NotImplemented()