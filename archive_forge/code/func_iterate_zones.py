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
def iterate_zones(self):
    offset = 0
    limit = 100
    while True:
        params = {'limit': limit, 'offset': offset}
        response = self.connection.request(action='/domains', params=params).object
        zones_list = response['domains']
        for item in zones_list:
            yield self._to_zone(item)
        if _rackspace_result_has_more(response, len(zones_list), limit):
            offset += limit
        else:
            break