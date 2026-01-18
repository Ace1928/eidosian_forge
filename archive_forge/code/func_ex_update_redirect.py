from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
def ex_update_redirect(self, redirect, redirect_to=None, name=None, type=None, iframe=None, query=None):
    """
        :param redirect: Record to update
        :type id: :class:`Redirect`

        :param redirect_to: The data field. (optional).
        :type redirect_to: ``str``

        :param name: The FQDN for the record.
        :type name: ``str``

        :param type: The type of redirects 301, 302 or 0 for iframes.
                     (optional).
        :type type: ``str``

        :param iframe: Title of iframe (optional).
        :type iframe: ``str``

        :param query: boolean Information about including query string when
                      redirecting. (optional).
        :type query: ``bool``

        :rtype: ``list`` of :class:`Redirect`
        """
    zone_id = redirect.zone.id
    r_json = {}
    if redirect_to is not None:
        r_json['redirect_to'] = redirect_to
    if name is not None:
        r_json['name'] = name
    if type is not None:
        r_json['record_type'] = type
    if iframe is not None:
        r_json['iframe_title'] = iframe
    if query is not None:
        r_json['redirect_query_string'] = query
    r_data = json.dumps({'zone_redirect': r_json})
    try:
        response = self.connection.request('/zones/{}/redirects/{}'.format(zone_id, redirect.id), method='PUT', data=r_data)
    except (BaseHTTPError, MalformedResponseError) as e:
        if isinstance(e, MalformedResponseError) and e.body == 'Not found':
            raise PointDNSException(value="Couldn't found redirect", http_code=httplib.NOT_FOUND, driver=self)
        raise PointDNSException(value=e.message, http_code=e.code, driver=self)
    redirect = self._to_redirect(response.object, zone=redirect.zone)
    return redirect