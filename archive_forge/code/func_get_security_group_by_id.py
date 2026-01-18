from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
def get_security_group_by_id(self, id):
    """Get a security group by ID

        :param id: ID of the security group.
        :returns: A security group
            ``openstack.network.v2.security_group.SecurityGroup``.
        """
    if not self._has_secgroups():
        raise exc.OpenStackCloudUnavailableFeature('Unavailable feature: security groups')
    error_message = f'Error getting security group with ID {id}'
    if self._use_neutron_secgroups():
        return self.network.get_security_group(id)
    else:
        data = proxy._json_response(self.compute.get(f'/os-security-groups/{id}'), error_message=error_message)
    return self._normalize_secgroup(self._get_and_munchify('security_group', data))