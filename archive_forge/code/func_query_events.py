import json
import logging as log
from urllib import parse as urlparse
import netaddr
from oslo_concurrency.lockutils import synchronized
import requests
from osprofiler.drivers import base
from osprofiler import exc
def query_events(self, params):
    constraints = []
    for field, value in params.items():
        constraints.append('%s/CONTAINS+%s' % (field, value))
    constraints.append('timestamp/GT+0')
    path = '%s/%s' % (self.QUERY_EVENTS_BASE_PATH, '/'.join(constraints))

    def _query_events():
        return self._send_request('get', 'https', path, headers=self._get_auth_header(), params={'limit': 20000, 'timeout': self._query_timeout})
    try:
        resp = _query_events()
    except exc.LogInsightLoginTimeout:
        LOG.debug('Current session timed out.')
        self.login()
        resp = _query_events()
    return resp