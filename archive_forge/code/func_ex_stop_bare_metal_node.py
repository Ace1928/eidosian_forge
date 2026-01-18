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
def ex_stop_bare_metal_node(self, node: Node) -> bool:
    """Stop the given bare metal node.

        :param node: The bare metal node to be stopped.
        :type node: :class: `Node`

        :rtype: ``bool``
        """
    resp = self.connection.request('/v2/bare-metals/%s/halt' % node.id, method='POST')
    return resp.success()