from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
def ex_list_mail_redirects(self, zone):
    """
        :param zone: Zone to list redirects for.
        :type zone: :class:`Zone`

        :rtype: ``list`` of :class:`MailRedirect`
        """
    response = self.connection.request('/zones/%s/mail_redirects' % zone.id)
    mail_redirects = self._to_mail_redirects(response.object, zone)
    return mail_redirects