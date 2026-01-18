import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
def get_active_server(self, server, auto_ip=True, ips=None, ip_pool=None, reuse=True, wait=False, timeout=180, nat_destination=None):
    if server['status'] == 'ERROR':
        if 'fault' in server and server['fault'] is not None and ('message' in server['fault']):
            raise exceptions.SDKException('Error in creating the server. Compute service reports fault: {reason}'.format(reason=server['fault']['message']), extra_data=dict(server=server))
        raise exceptions.SDKException('Error in creating the server (no further information available)', extra_data=dict(server=server))
    if server['status'] == 'ACTIVE':
        if 'addresses' in server and server['addresses']:
            return self.add_ips_to_server(server, auto_ip, ips, ip_pool, reuse=reuse, nat_destination=nat_destination, wait=wait, timeout=timeout)
        self.log.debug('Server %(server)s reached ACTIVE state without being allocated an IP address. Deleting server.', {'server': server['id']})
        try:
            self._delete_server(server=server, wait=wait, timeout=timeout)
        except Exception as e:
            raise exceptions.SDKException('Server reached ACTIVE state without being allocated an IP address AND then could not be deleted: {0}'.format(e), extra_data=dict(server=server))
        raise exceptions.SDKException('Server reached ACTIVE state without being allocated an IP address.', extra_data=dict(server=server))
    return None