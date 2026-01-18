from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def floating_ip_pool_list(self):
    """Get floating IP pools

        https://docs.openstack.org/api-ref/compute/?expanded=#list-floating-ip-pools

        :returns:
            list of floating IP pools
        """
    url = '/os-floating-ip-pools'
    return self.list(url)['floating_ip_pools']