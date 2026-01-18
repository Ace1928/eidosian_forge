import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def ex_describe_attachment(self, attachment_id):
    path = '/metal/v1/storage/attachments/%s' % attachment_id
    data = self.connection.request(path).object
    return data