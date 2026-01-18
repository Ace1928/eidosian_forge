from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def runtime_version(self):
    """Return runtime version of the TPU."""
    if not self._use_api:
        url = _VERSION_SWITCHER_ENDPOINT.format(self.network_endpoints()[0]['ipAddress'])
        try:
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req)
            version_details = json.loads(resp.read())
            return version_details.get('currentVersion')
        except urllib.error.HTTPError as e:
            status_code = e.code
            if status_code == 404:
                return None
            else:
                raise e
    return self._get_tpu_property('tensorflowVersion')