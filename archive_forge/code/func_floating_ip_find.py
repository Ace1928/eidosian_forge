from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def floating_ip_find(self, floating_ip=None):
    """Return a security group given name or ID

        https://docs.openstack.org/api-ref/compute/#list-floating-ip-addresses

        :param string floating_ip:
            Floating IP address
        :returns: A dict of the floating IP attributes
        """
    url = '/os-floating-ips'
    return self.find(url, attr='ip', value=floating_ip)