from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError
def ex_check_availability(self, domain, for_transfer=False):
    """
        Check the availability of the domain

        :param   domain: the domain name e.g. wazzlewobbleflooble.com
        :type    domain: ``str``

        :param   for_transfer: Check if domain is available for transfer
        :type    for_transfer: ``bool``

        :rtype: `list` of :class:`GoDaddyAvailability`
        """
    result = self.connection.request('/v1/domains/available', method='GET', params={'domain': domain, 'forTransfer': str(for_transfer)}).object
    return GoDaddyAvailability(domain=result['domain'], available=result['available'], price=result['price'], currency=result['currency'], period=result['period'])