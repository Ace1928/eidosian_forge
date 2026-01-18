import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def get_service_names(self, service_type=None, region=None):
    """
        Retrieve list of service names that match service type and region.

        :type service_type: ``str``
        :type region: ``str``

        :rtype: ``list`` of ``str``
        """
    names = set()
    if '2.0' not in self._auth_version:
        raise ValueError('Unsupported version: %s' % self._auth_version)
    for entry in self._entries:
        if service_type and entry.service_type != service_type:
            continue
        include = True
        for endpoint in entry.endpoints:
            if region and endpoint.region != region:
                include = False
                break
        if include and entry.service_name:
            names.add(entry.service_name)
    return sorted(list(names))