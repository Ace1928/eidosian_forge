from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
def ex_create_mail_redirect(self, destination, source, zone):
    """
        :param destination: The destination address of mail redirect.
        :type destination: ``str``

        :param source: The source address of mail redirect.
        :type source: ``str``

        :param zone: Zone to list redirects for.
        :type zone: :class:`Zone`

        :rtype: ``list`` of :class:`MailRedirect`
        """
    r_json = {'destination_address': destination, 'source_address': source}
    r_data = json.dumps({'zone_mail_redirect': r_json})
    try:
        response = self.connection.request('/zones/%s/mail_redirects' % zone.id, method='POST', data=r_data)
    except (BaseHTTPError, MalformedResponseError) as e:
        raise PointDNSException(value=e.message, http_code=e.code, driver=self)
    mail_redirect = self._to_mail_redirect(response.object, zone=zone)
    return mail_redirect