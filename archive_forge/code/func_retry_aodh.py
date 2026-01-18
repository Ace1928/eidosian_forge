import os
import time
import os_client_config
from oslo_utils import uuidutils
from tempest.lib.cli import base
from tempest.lib import exceptions
def retry_aodh(self, retry, *args, **kwargs):
    result = ''
    while not result.strip() and retry > 0:
        result = self.aodh(*args, **kwargs)
        if not result:
            time.sleep(1)
            retry -= 1
    return result