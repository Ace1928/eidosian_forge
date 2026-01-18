from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError
def ex_list_tlds(self):
    """
        List available TLDs for sale

        :rtype: ``list`` of :class:`GoDaddyTLD`
        """
    result = self.connection.request('/v1/domains/tlds', method='GET').object
    return self._to_tlds(result)