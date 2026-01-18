from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
def ex_get_mail_redirects(self, zone_id, mail_r_id):
    """
        :param zone: Zone to list redirects for.
        :type zone: :class:`Zone`

        :param mail_r_id: Mail redirect id.
        :type mail_r_id: ``str``

        :rtype: ``list`` of :class:`MailRedirect`
        """
    try:
        response = self.connection.request('/zones/{}/mail_redirects/{}'.format(zone_id, mail_r_id))
    except (BaseHTTPError, MalformedResponseError) as e:
        if isinstance(e, MalformedResponseError) and e.body == 'Not found':
            raise PointDNSException(value="Couldn't found mail redirect", http_code=httplib.NOT_FOUND, driver=self)
        raise PointDNSException(value=e.message, http_code=e.code, driver=self)
    mail_redirect = self._to_mail_redirect(response.object, zone_id=zone_id)
    return mail_redirect