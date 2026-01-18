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
def ex_list_bare_metal_sizes(self) -> List[NodeSize]:
    """List bare metal sizes.

        :rtype: ``list`` of :class: `NodeSize`
        """
    data = self._paginated_request('/v2/plans-metal', 'plans_metal')
    return [self._to_size(item) for item in data]