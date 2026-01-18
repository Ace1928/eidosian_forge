import abc
from keystone import exception
@abc.abstractmethod
def list_protocols(self, idp_id):
    """List an IdP's supported protocols.

        :param idp_id: ID of IdP object
        :type idp_id: string
        :raises keystone.exception.IdentityProviderNotFound: If the IdP
            doesn't exist.
        :returns: list of protocol ref
        :rtype: list of dict

        """
    raise exception.NotImplemented()