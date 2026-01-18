from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def network_endpoints(self):
    """Return a list of tpu endpoints."""
    if not self._use_api:
        return list(_environment_var_to_network_endpoints(self._tpu))
    response = self._fetch_cloud_tpu_metadata()
    if response.get('state') != 'READY':
        raise RuntimeError('TPU "%s" is not yet ready; state: "%s"' % (self._tpu, response.get('state')))
    if 'networkEndpoints' in response:
        return response['networkEndpoints']
    else:
        return [{'ipAddress': response['ipAddress'], 'port': response['port']}]