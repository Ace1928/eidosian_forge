from manilaclient import api_versions
from manilaclient import base
from manilaclient import exceptions
def remove_security_service(self, share_network, security_service):
    """Dissociate security service from a share network.

        :param share_network: share network name, id or ShareNetwork instance
        :param security_service: name, id or SecurityService instance
        :rtype: :class:`ShareNetwork`
        """
    body = {'remove_security_service': {'security_service_id': base.getid(security_service)}}
    return self._create(RESOURCE_PATH % base.getid(share_network) + '/action', body, RESOURCE_NAME)