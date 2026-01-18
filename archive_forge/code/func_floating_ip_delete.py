from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def floating_ip_delete(self, floating_ip_id=None):
    """Delete a floating IP

        https://docs.openstack.org/api-ref/compute/#delete-deallocate-floating-ip-address

        :param string floating_ip_id:
            Floating IP ID
        """
    url = '/os-floating-ips'
    if floating_ip_id is not None:
        return self.delete('/%s/%s' % (url, floating_ip_id))
    return None