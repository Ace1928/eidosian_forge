from manilaclient import api_versions
from manilaclient import base
from manilaclient import exceptions
def add_security_service(self, share_network, security_service):
    """Associate given security service with a share network.

        :param share_network: share network name, id or ShareNetwork instance
        :param security_service: name, id or SecurityService instance
        :rtype: :class:`ShareNetwork`
        """
    info = {'security_service_id': base.getid(security_service)}
    return self._action('add_security_service', share_network, info)