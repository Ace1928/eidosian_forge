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
def find_custom_field_key(self, key_id):
    """Return custom field key name, provided it's id"""
    if not hasattr(self, 'custom_fields'):
        content = self.connection.RetrieveContent()
        if content.customFieldsManager:
            self.custom_fields = content.customFieldsManager.field
        else:
            self.custom_fields = []
    for k in self.custom_fields:
        if k.key == key_id:
            return k.name
    return None