import json
import time
import base64
from typing import Any, Dict, List, Union, Optional
from functools import update_wrapper
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, InvalidCredsError, ServiceUnavailableError
from libcloud.common.vultr import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.utils.publickey import get_pubkey_openssh_fingerprint
def require_api_key(self):
    """
        Check whether this call (method + action) must be authenticated.

        :return: True if ``API-Key`` header required, False otherwise.
        :rtype: bool
        """
    try:
        return self.method not in self.unauthenticated_endpoints[self.action]
    except KeyError:
        return True