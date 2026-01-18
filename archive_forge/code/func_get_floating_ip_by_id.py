import ipaddress
import time
import warnings
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
from openstack import warnings as os_warnings
def get_floating_ip_by_id(self, id):
    """Get a floating ip by ID

        :param id: ID of the floating ip.
        :returns: A floating ip
            `:class:`~openstack.network.v2.floating_ip.FloatingIP`.
        """
    error_message = 'Error getting floating ip with ID {id}'.format(id=id)
    if self._use_neutron_floating():
        fip = self.network.get_ip(id)
        return fip
    else:
        data = proxy._json_response(self.compute.get('/os-floating-ips/{id}'.format(id=id)), error_message=error_message)
        return self._normalize_floating_ip(self._get_and_munchify('floating_ip', data))