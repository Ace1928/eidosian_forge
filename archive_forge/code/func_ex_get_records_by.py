import re
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.nfsn import NFSNConnection
from libcloud.common.exceptions import BaseHTTPError
def ex_get_records_by(self, zone, name=None, type=None):
    """
        Return a list of records for the provided zone, filtered by name and/or
        type.

        :param zone: Zone to list records for.
        :type zone: :class:`Zone`

        :param zone: Zone where the requested records are found.
        :type  zone: :class:`Zone`

        :param name: name of the records, for example "www". (optional)
        :type  name: ``str``

        :param type: DNS record type (A, MX, TXT). (optional)
        :type  type: :class:`RecordType`

        :return: ``list`` of :class:`Record`
        """
    payload = {}
    if name is not None:
        payload['name'] = name
    if type is not None:
        payload['type'] = type
    action = '/dns/%s/listRRs' % zone.domain
    response = self.connection.request(action=action, data=payload, method='POST')
    return self._to_records(response, zone)