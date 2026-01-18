from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
def ex_get_redirect(self, zone_id, redirect_id):
    """
        :param zone: Zone to list redirects for.
        :type zone: :class:`Zone`

        :param redirect_id: Redirect id.
        :type redirect_id: ``str``

        :rtype: ``list`` of :class:`Redirect`
        """
    try:
        response = self.connection.request('/zones/{}/redirects/{}'.format(zone_id, redirect_id))
    except (BaseHTTPError, MalformedResponseError) as e:
        if isinstance(e, MalformedResponseError) and e.body == 'Not found':
            raise PointDNSException(value="Couldn't found redirect", http_code=httplib.NOT_FOUND, driver=self)
        raise PointDNSException(value=e.message, http_code=e.code, driver=self)
    redirect = self._to_redirect(response.object, zone_id=zone_id)
    return redirect