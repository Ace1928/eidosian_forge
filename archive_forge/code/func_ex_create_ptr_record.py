import copy
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import get_new_obj, merge_valid_keys
from libcloud.common.base import PollingConnection
from libcloud.common.types import LibcloudError
from libcloud.common.openstack import OpenStackDriverMixin
from libcloud.common.rackspace import AUTH_URL
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.drivers.openstack import OpenStack_1_1_Response, OpenStack_1_1_Connection
def ex_create_ptr_record(self, device, ip, domain, extra=None):
    """
        Create a PTR record for a specific IP on a specific device.

        The ``device`` should be an instance of one of these:
            :class:`libcloud.compute.base.Node`
            :class:`libcloud.loadbalancer.base.LoadBalancer`

        And it needs to have the following ``extra`` fields set:
            service_name - the service catalog name for the device
            uri - the URI pointing to the GET endpoint for the device

        Those are automatically set for you if you got the device from
        the Rackspace driver for that service.

        For example:
            server = rs_compute.ex_get_node_details(id)
            rs_dns.create_ptr_record(server, ip, domain)

            loadbalancer = rs_lbs.get_balancer(id)
            rs_dns.create_ptr_record(loadbalancer, ip, domain)

        :param device: the device that owns the IP
        :param ip: the IP for which you want to set reverse DNS
        :param domain: the fqdn you want that IP to represent
        :param extra: a ``dict`` with optional extra values:
            ttl - the time-to-live of the PTR record
        :rtype: instance of :class:`RackspacePTRRecord`
        """
    _check_ptr_extra_fields(device)
    if extra is None:
        extra = {}
    data = {'name': domain, 'type': RecordType.PTR, 'data': ip}
    if 'ttl' in extra:
        data['ttl'] = extra['ttl']
    payload = {'recordsList': {'records': [data]}, 'link': {'content': '', 'href': device.extra['uri'], 'rel': device.extra['service_name']}}
    response = self.connection.async_request(action='/rdns', method='POST', data=payload).object
    item = next(iter(response['response']['records']))
    return self._to_ptr_record(data=item, link=payload['link'])