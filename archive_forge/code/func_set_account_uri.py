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
def set_account_uri(self, uri):
    """
        Set account URI. For ACME v2, it needs to be used to sending signed
        requests.
        """
    self.account_uri = uri
    if self.version != 1:
        self.account_jws_header.pop('jwk')
        self.account_jws_header['kid'] = self.account_uri