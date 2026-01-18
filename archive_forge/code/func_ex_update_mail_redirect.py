from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
def ex_update_mail_redirect(self, mail_r, destination, source=None):
    """
        :param mail_r: Mail redirect to update
        :type mail_r: :class:`MailRedirect`

        :param destination: The destination address of mail redirect.
        :type destination: ``str``

        :param source: The source address of mail redirect. (optional)
        :type source: ``str``

        :rtype: ``list`` of :class:`MailRedirect`
        """
    zone_id = mail_r.zone.id
    r_json = {'destination_address': destination}
    if source is not None:
        r_json['source_address'] = source
    r_data = json.dumps({'zone_redirect': r_json})
    try:
        response = self.connection.request('/zones/{}/mail_redirects/{}'.format(zone_id, mail_r.id), method='PUT', data=r_data)
    except (BaseHTTPError, MalformedResponseError) as e:
        if isinstance(e, MalformedResponseError) and e.body == 'Not found':
            raise PointDNSException(value="Couldn't found mail redirect", http_code=httplib.NOT_FOUND, driver=self)
        raise PointDNSException(value=e.message, http_code=e.code, driver=self)
    mail_redirect = self._to_mail_redirect(response.object, zone=mail_r.zone)
    return mail_redirect