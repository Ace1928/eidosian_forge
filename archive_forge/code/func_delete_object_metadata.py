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
def delete_object_metadata(self, obj, container=None, keys=None):
    """Delete metadata for an object.

        :param obj: The value can be the name of an object or a
            :class:`~openstack.object_store.v1.obj.Object` instance.
        :param container: The value can be the ID of a container or a
            :class:`~openstack.object_store.v1.container.Container` instance.
        :param keys: The keys of metadata to be deleted.
        """
    container_name = self._get_container_name(obj, container)
    res = self._get_resource(_obj.Object, obj, container=container_name)
    res.delete_metadata(self, keys)
    return res