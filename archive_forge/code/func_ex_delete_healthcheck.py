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
def ex_delete_healthcheck(self, healthcheck):
    """
        Remove an existing Health Check

        :param zone: The healthcheck which has to be removed
        :type zone: :class:`AuroraDNSHealthCheck`
        """
    self.connection.set_context({'resource': 'healthcheck', 'id': healthcheck.id})
    self.connection.request('/zones/{}/health_checks/{}'.format(healthcheck.zone.id, healthcheck.id), method='DELETE')
    return True