import abc
from keystone import exception
@abc.abstractmethod
def delete_idp(self, idp_id):
    """Delete an identity provider.

        :param idp_id: ID of IdP object
        :type idp_id: string
        :raises keystone.exception.IdentityProviderNotFound: If the IdP
            doesn't exist.

        """
    raise exception.NotImplemented()