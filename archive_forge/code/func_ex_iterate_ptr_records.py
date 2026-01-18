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
def ex_iterate_ptr_records(self, device):
    """
        Return a generator to iterate over existing PTR Records.

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
            ptr_iter = rs_dns.ex_list_ptr_records(server)

            loadbalancer = rs_lbs.get_balancer(id)
            ptr_iter = rs_dns.ex_list_ptr_records(loadbalancer)

        Note: the Rackspace DNS API docs indicate that the device 'href' is
        optional, but testing does not bear this out. It throws a
        400 Bad Request error if you do not pass in the 'href' from
        the server or loadbalancer.  So ``device`` is required.

        :param device: the device that owns the IP
        :rtype: ``generator`` of :class:`RackspacePTRRecord`
        """
    _check_ptr_extra_fields(device)
    params = {'href': device.extra['uri']}
    service_name = device.extra['service_name']
    self.connection.set_context({'resource': 'ptr_records'})
    try:
        response = self.connection.request(action='/rdns/%s' % service_name, params=params).object
        records = response['records']
        link = dict(rel=service_name, **params)
        for item in records:
            record = self._to_ptr_record(data=item, link=link)
            yield record
    except BaseHTTPError as exc:
        if exc.code == 404:
            return
        raise