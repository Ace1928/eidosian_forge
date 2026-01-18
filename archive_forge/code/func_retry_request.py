import inspect
import itertools
import logging
import re
import time
import urllib.parse as urlparse
import debtcollector.renames
from keystoneauth1 import exceptions as ksa_exc
import requests
from neutronclient._i18n import _
from neutronclient import client
from neutronclient.common import exceptions
from neutronclient.common import extension as client_extension
from neutronclient.common import serializer
from neutronclient.common import utils
def retry_request(self, method, action, body=None, headers=None, params=None):
    """Call do_request with the default retry configuration.

        Only idempotent requests should retry failed connection attempts.
        :raises: ConnectionFailed if the maximum # of retries is exceeded
        """
    max_attempts = self.retries + 1
    for i in range(max_attempts):
        try:
            return self.do_request(method, action, body=body, headers=headers, params=params)
        except (exceptions.ConnectionFailed, ksa_exc.ConnectionError):
            if i < self.retries:
                _logger.debug('Retrying connection to Neutron service')
                time.sleep(self.retry_interval)
            elif self.raise_errors:
                raise
    if self.retries:
        msg = _('Failed to connect to Neutron server after %d attempts') % max_attempts
    else:
        msg = _('Failed to connect Neutron server')
    raise exceptions.ConnectionFailed(reason=msg)