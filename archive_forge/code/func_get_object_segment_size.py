from calendar import timegm
import collections
from hashlib import sha1
import hmac
import json
import os
import time
from urllib import parse
from openstack import _log
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1 import account as _account
from openstack.object_store.v1 import container as _container
from openstack.object_store.v1 import info as _info
from openstack.object_store.v1 import obj as _obj
from openstack import proxy
from openstack import utils
def get_object_segment_size(self, segment_size):
    """Get a segment size that will work given capabilities"""
    if segment_size is None:
        segment_size = DEFAULT_OBJECT_SEGMENT_SIZE
    min_segment_size = 0
    try:
        caps = self.get_info()
    except (exceptions.NotFoundException, exceptions.PreconditionFailedException):
        server_max_file_size = DEFAULT_MAX_FILE_SIZE
        self._connection.log.info('Swift capabilities not supported. Using default max file size.')
    except exceptions.SDKException:
        raise
    else:
        server_max_file_size = caps.swift.get('max_file_size', 0)
        min_segment_size = caps.slo.get('min_segment_size', 0)
    if segment_size > server_max_file_size:
        return server_max_file_size
    if segment_size < min_segment_size:
        return min_segment_size
    return segment_size