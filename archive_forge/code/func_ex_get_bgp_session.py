import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_get_bgp_session(self, session_uuid):
    path = '/metal/v1/bgp/sessions/%s' % session_uuid
    return self.connection.request(path).object