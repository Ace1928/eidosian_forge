from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def configure_tpu_version(self, version, restart_type='always'):
    """Configure TPU software version.

    Args:
      version (string): Version of software to configure the TPU with.
      restart_type (string): Restart behaviour when switching versions,
        defaults to always restart. Options are 'always', 'ifNeeded'.

    """

    def configure_worker(worker):
        """Configure individual TPU worker.

      Args:
        worker: A dict with the field ipAddress where the configure request will
          be sent.
      """
        ip_address = worker['ipAddress']
        url = (_VERSION_SWITCHER_ENDPOINT + '/{}?restartType={}').format(ip_address, version, restart_type)
        req = urllib.request.Request(url, data=b'')
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            status_code = e.code
            if status_code == 404:
                raise Exception('Tensorflow version {} is not available on Cloud TPU, try a previous nightly version or refer to https://cloud.google.com/tpu/docs/release-notes for the latest official version.'.format(version))
            else:
                raise Exception('Failed to configure worker {}'.format(ip_address))
    workers = self.network_endpoints()
    with futures.ThreadPoolExecutor(max_workers=len(workers)) as executor:
        results = executor.map(configure_worker, workers)
        for result in results:
            if result:
                result.result()