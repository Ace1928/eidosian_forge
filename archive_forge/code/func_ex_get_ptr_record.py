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
def ex_get_ptr_record(self, service_name, record_id):
    """
        Get a specific PTR record by id.

        :param service_name: the service catalog name of the linked device(s)
                             i.e. cloudLoadBalancers or cloudServersOpenStack
        :param record_id: the id (i.e. PTR-12345) of the PTR record
        :rtype: instance of :class:`RackspacePTRRecord`
        """
    self.connection.set_context({'resource': 'record', 'id': record_id})
    response = self.connection.request(action='/rdns/{}/{}'.format(service_name, record_id)).object
    item = next(iter(response['recordsList']['records']))
    return self._to_ptr_record(data=item, link=response['link'])