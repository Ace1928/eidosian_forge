import abc
from keystone import exception
@abc.abstractmethod
def list_idps(self, hints):
    """List all identity providers.

        :param hints: filter hints which the driver should
                      implement if at all possible.
        :returns: list of idp refs
        :rtype: list of dicts

        :raises keystone.exception.IdentityProviderNotFound: If the IdP
            doesn't exist.

        """
    raise exception.NotImplemented()