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
def list_floating_ips(self, filters=None):
    """List all available floating IPs.

        :param filters: (optional) dict of filter conditions to push down
        :returns: A list of floating IP
            ``openstack.network.v2.floating_ip.FloatingIP``.
        """
    if not filters:
        filters = {}
    if self._use_neutron_floating():
        try:
            return self._neutron_list_floating_ips(filters)
        except exceptions.NotFoundException as e:
            if filters:
                self.log.error("Neutron returned NotFound for floating IPs, which means this cloud doesn't have neutron floating ips. openstacksdk can't fallback to trying Nova since nova doesn't support server-side filtering when listing floating ips and filters were given. If you do not think openstacksdk should be attempting to list floating IPs on neutron, it is possible to control the behavior by setting floating_ip_source to 'nova' or None for cloud %(cloud)r in 'clouds.yaml'.", {'cloud': self.name})
                return []
            self.log.debug("Something went wrong talking to neutron API: '%(msg)s'. Trying with Nova.", {'msg': str(e)})
    elif filters:
        raise ValueError("Nova-network don't support server-side floating ips filtering. Use the search_floating_ips method instead")
    floating_ips = self._nova_list_floating_ips()
    return self._normalize_floating_ips(floating_ips)