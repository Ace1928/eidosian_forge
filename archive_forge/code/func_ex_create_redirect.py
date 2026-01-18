from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
def ex_create_redirect(self, redirect_to, name, type, zone, iframe=None, query=None):
    """
        :param redirect_to: The data field. (redirect_to)
        :type redirect_to: ``str``

        :param name: The FQDN for the record.
        :type name: ``str``

        :param type: The type of redirects 301, 302 or 0 for iframes.
        :type type: ``str``

        :param zone: Zone to list redirects for.
        :type zone: :class:`Zone`

        :param iframe: Title of iframe (optional).
        :type iframe: ``str``

        :param query: boolean Information about including query string when
                      redirecting. (optional).
        :type query: ``bool``

        :rtype: :class:`Record`
        """
    r_json = {'name': name, 'redirect_to': redirect_to}
    if type is not None:
        r_json['redirect_type'] = type
    if iframe is not None:
        r_json['iframe_title'] = iframe
    if query is not None:
        r_json['redirect_query_string'] = query
    r_data = json.dumps({'zone_redirect': r_json})
    try:
        response = self.connection.request('/zones/%s/redirects' % zone.id, method='POST', data=r_data)
    except (BaseHTTPError, MalformedResponseError) as e:
        raise PointDNSException(value=e.message, http_code=e.code, driver=self)
    redirect = self._to_redirect(response.object, zone=zone)
    return redirect