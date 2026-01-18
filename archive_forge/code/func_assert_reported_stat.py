import itertools
import os
import pprint
import select
import socket
import threading
import time
import fixtures
from keystoneauth1 import exceptions
import prometheus_client
from requests import exceptions as rexceptions
import testtools.content
from openstack.tests.unit import base
def assert_reported_stat(self, key, value=None, kind=None):
    """Check statsd output

        Check statsd return values.  A ``value`` should specify a
        ``kind``, however a ``kind`` may be specified without a
        ``value`` for a generic match.  Leave both empy to just check
        for key presence.

        :arg str key: The statsd key
        :arg str value: The expected value of the metric ``key``
        :arg str kind: The expected type of the metric ``key``  For example

          - ``c`` counter
          - ``g`` gauge
          - ``ms`` timing
          - ``s`` set
        """
    self.assertIsNotNone(self.statsd)
    if value:
        self.assertNotEqual(kind, None)
    start = time.time()
    while time.time() < start + 1:
        stats = itertools.chain.from_iterable([s.decode('utf-8').split('\n') for s in self.statsd.stats])
        for stat in stats:
            k, v = stat.split(':')
            if key == k:
                if kind is None:
                    return True
                s_value, s_kind = v.split('|')
                if kind != s_kind:
                    continue
                if value:
                    if kind == 'ms':
                        if float(value) == float(s_value):
                            return True
                    if value == s_value:
                        return True
                    continue
                return True
        time.sleep(0.1)
    raise Exception('Key %s not found in reported stats' % key)