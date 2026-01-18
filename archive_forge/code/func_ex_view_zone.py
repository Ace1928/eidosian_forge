import re
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.types import LibcloudError
from libcloud.common.worldwidedns import WorldWideDNSConnection
def ex_view_zone(self, domain, name_server):
    """
        View zone file from a name server

        :param domain: Domain name.
        :type  domain: ``str``

        :param name_server: Name server to check. (1, 2 or 3)
        :type  name_server: ``int``

        :rtype: ``str``

        For more info, please see:
        https://www.worldwidedns.net/dns_api_protocol_viewzone.asp
        or
        https://www.worldwidedns.net/dns_api_protocol_viewzone_reseller.asp
        """
    params = {'DOMAIN': domain, 'NS': name_server}
    action = '/api_dns_viewzone.asp'
    if self.reseller_id is not None:
        action = '/api_dns_viewzone_reseller.asp'
    response = self.connection.request(action, params=params)
    return response.object