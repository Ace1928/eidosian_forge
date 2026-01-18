import hmac
import json
import base64
import datetime
from hashlib import sha256
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
def ex_list_healthchecks(self, zone):
    """
        List all Health Checks in a zone.

        :param zone: Zone to list health checks for.
        :type zone: :class:`Zone`

        :return: ``list`` of :class:`AuroraDNSHealthCheck`
        """
    healthchecks = []
    self.connection.set_context({'resource': 'zone', 'id': zone.id})
    res = self.connection.request('/zones/%s/health_checks' % zone.id)
    for healthcheck in res.parse_body():
        healthchecks.append(self.__res_to_healthcheck(zone, healthcheck))
    return healthchecks