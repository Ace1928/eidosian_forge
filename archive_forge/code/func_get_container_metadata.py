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
def get_container_metadata(self, container):
    """Get metadata for a container

        :param container: The value can be the name of a container or a
            :class:`~openstack.object_store.v1.container.Container` instance.

        :returns: One :class:`~openstack.object_store.v1.container.Container`
        :raises: :class:`~openstack.exceptions.ResourceNotFound`
            when no resource can be found.
        """
    return self._head(_container.Container, container)