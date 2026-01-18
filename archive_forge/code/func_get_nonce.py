from __future__ import absolute_import, division, print_function
import copy
import datetime
import json
import locale
import time
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import PY3
from ansible_collections.community.crypto.plugins.module_utils.acme.backend_openssl_cli import (
from ansible_collections.community.crypto.plugins.module_utils.acme.backend_cryptography import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
def get_nonce(self, resource=None):
    url = self.directory_root if self.version == 1 else self.directory['newNonce']
    if resource is not None:
        url = resource
    retry_count = 0
    while True:
        response, info = fetch_url(self.module, url, method='HEAD', timeout=self.request_timeout)
        if _decode_retry(self.module, response, info, retry_count):
            retry_count += 1
            continue
        if info['status'] not in (200, 204):
            raise NetworkException('Failed to get replay-nonce, got status {0}'.format(format_http_status(info['status'])))
        if 'replay-nonce' in info:
            return info['replay-nonce']
        self.module.log('HEAD to {0} did return status {1}, but no replay-nonce header!'.format(url, format_http_status(info['status'])))
        if retry_count >= 5:
            raise ACMEProtocolException(self.module, msg='Was not able to obtain nonce, giving up after 5 retries', info=info, response=response)
        retry_count += 1