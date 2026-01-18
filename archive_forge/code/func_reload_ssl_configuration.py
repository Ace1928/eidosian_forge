from __future__ import absolute_import, division, print_function
import binascii
import random
import re
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def reload_ssl_configuration(self):
    """Asynchronously reloads the SSL configuration."""
    self.request(self.url_path_prefix + 'certificates/reload%s' % self.url_path_suffix, method='POST', ignore_errors=True)
    for retry in range(int(self.RESET_SSL_CONFIG_TIMEOUT_SEC / 3)):
        try:
            rc, current_certificates = self.request(self.url_path_prefix + 'certificates/server%s' % self.url_path_suffix)
        except Exception as error:
            sleep(3)
            continue
        break
    else:
        self.module.fail_json(msg='Failed to retrieve server certificates. Array [%s].' % self.ssid)