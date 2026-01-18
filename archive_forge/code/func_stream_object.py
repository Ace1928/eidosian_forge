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
def stream_object(self, obj, container=None, chunk_size=1024, **attrs):
    """Stream the data contained inside an object.

        :param obj: The value can be the name of an object or a
            :class:`~openstack.object_store.v1.obj.Object` instance.
        :param container: The value can be the name of a container or a
            :class:`~openstack.object_store.v1.container.Container` instance.

        :raises: :class:`~openstack.exceptions.ResourceNotFound`
            when no resource can be found.
        :returns: An iterator that iterates over chunk_size bytes
        """
    container_name = self._get_container_name(obj=obj, container=container)
    obj = self._get_resource(_obj.Object, obj, container=container_name, **attrs)
    return obj.stream(self, chunk_size=chunk_size)