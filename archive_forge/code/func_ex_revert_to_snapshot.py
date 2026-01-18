import ssl
import json
import time
import atexit
import base64
import asyncio
import hashlib
import logging
import warnings
import functools
import itertools
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def ex_revert_to_snapshot(self, node, snapshot_name=None):
    """
        Revert node to a specific snapshot.
        If snapshot_name is not defined revert to the last one.
        """
    if self.driver_soap is None:
        self._get_soap_driver()
    return self.driver_soap.ex_revert_to_snapshot(node, snapshot_name=snapshot_name)