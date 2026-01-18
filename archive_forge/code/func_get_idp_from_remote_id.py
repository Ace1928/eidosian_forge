import abc
from keystone import exception
@abc.abstractmethod
def get_idp_from_remote_id(self, remote_id):
    """Get an identity provider by remote ID.

        :param remote_id: ID of remote IdP
        :type idp_id: string
        :raises keystone.exception.IdentityProviderNotFound: If the IdP
            doesn't exist.
        :returns: idp ref
        :rtype: dict

        """
    raise exception.NotImplemented()