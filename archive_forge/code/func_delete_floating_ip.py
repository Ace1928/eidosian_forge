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
def delete_floating_ip(self, floating_ip_id, retry=1):
    """Deallocate a floating IP from a project.

        :param floating_ip_id: a floating IP address ID.
        :param retry: number of times to retry. Optional, defaults to 1,
                      which is in addition to the initial delete call.
                      A value of 0 will also cause no checking of results to
                      occur.

        :returns: True if the IP address has been deleted, False if the IP
            address was not found.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    for count in range(0, max(0, retry) + 1):
        result = self._delete_floating_ip(floating_ip_id)
        if retry == 0 or not result:
            return result
        f_ip = self.get_floating_ip(id=floating_ip_id)
        if not f_ip or f_ip['status'] == 'DOWN':
            return True
    raise exceptions.SDKException('Attempted to delete Floating IP {ip} with ID {id} a total of {retry} times. Although the cloud did not indicate any errors the floating ip is still in existence. Aborting further operations.'.format(id=floating_ip_id, ip=f_ip['floating_ip_address'], retry=retry + 1))