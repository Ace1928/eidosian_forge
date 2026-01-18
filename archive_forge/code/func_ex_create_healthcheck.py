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
def ex_create_healthcheck(self, zone, type, hostname, port, path, interval, threshold, ipaddress=None, enabled=True, extra=None):
    """
        Create a new Health Check in a zone

        :param zone: Zone in which the health check should be created
        :type zone: :class:`Zone`

        :param type: The type of health check to be created
        :type  type: :class:`AuroraDNSHealthCheckType`

        :param hostname: The hostname of the target to monitor
        :type  hostname: ``str``

        :param port: The port of the target to monitor. E.g. 80 for HTTP
        :type  port: ``int``

        :param path: The path of the target to monitor. Only used by HTTP
                     at this moment. Usually this is simple /.
        :type  path: ``str``

        :param interval: The interval of checks. 10, 30 or 60 seconds.
        :type  interval: ``int``

        :param threshold: The threshold of failures before the healthcheck is
                          marked as failed.
        :type  threshold: ``int``

        :param ipaddress: (optional) The IP Address of the target to monitor.
                          You can pass a empty string if this is not required.
        :type  ipaddress: ``str``

        :param enabled: (optional) If this healthcheck is enabled to run
        :type  enabled: ``bool``

        :param extra: (optional) Extra attributes (driver specific).
        :type  extra: ``dict``

        :return: :class:`AuroraDNSHealthCheck`
        """
    cdata = {'type': self.HEALTHCHECK_TYPE_MAP[type], 'hostname': hostname, 'ipaddress': ipaddress, 'port': int(port), 'interval': int(interval), 'path': path, 'threshold': int(threshold), 'enabled': enabled}
    self.connection.set_context({'resource': 'zone', 'id': zone.id})
    res = self.connection.request('/zones/%s/health_checks' % zone.id, method='POST', data=json.dumps(cdata))
    healthcheck = res.parse_body()
    return self.__res_to_healthcheck(zone, healthcheck)