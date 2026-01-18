import abc
from keystone import exception
@abc.abstractmethod
def get_mapping_from_idp_and_protocol(self, idp_id, protocol_id):
    """Get mapping based on idp_id and protocol_id.

        :param idp_id: id of the identity provider
        :type idp_id: string
        :param protocol_id: id of the protocol
        :type protocol_id: string
        :raises keystone.exception.IdentityProviderNotFound: If the IdP
            doesn't exist.
        :raises keystone.exception.FederatedProtocolNotFound: If the federated
            protocol cannot be found.
        :returns: mapping ref
        :rtype: dict

        """
    raise exception.NotImplemented()