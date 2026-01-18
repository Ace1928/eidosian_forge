import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_clone_volume(self, volume, snapshot=None):
    path = '/metal/v1/storage/%s/clone' % volume.id
    if snapshot:
        path += '?snapshot_timestamp=%s' % snapshot.extra['timestamp']
    res = self.connection.request(path, method='POST')
    return res.status == httplib.NO_CONTENT