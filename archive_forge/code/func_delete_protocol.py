import abc
from keystone import exception
@abc.abstractmethod
def delete_protocol(self, idp_id, protocol_id):
    """Delete an IdP-Protocol configuration.

        :param idp_id: ID of IdP object
        :type idp_id: string
        :param protocol_id: ID of protocol object
        :type protocol_id: string
        :raises keystone.exception.IdentityProviderNotFound: If the IdP
            doesn't exist.
        :raises keystone.exception.FederatedProtocolNotFound: If the federated
            protocol cannot be found.

        """
    raise exception.NotImplemented()