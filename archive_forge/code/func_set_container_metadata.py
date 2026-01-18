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
def set_container_metadata(self, container, refresh=True, **metadata):
    """Set metadata for a container.

        :param container: The value can be the name of a container or a
            :class:`~openstack.object_store.v1.container.Container`
            instance.
        :param refresh: Flag to trigger refresh of container object re-fetch.
        :param kwargs metadata: Key/value pairs to be set as metadata on the
            container. Both custom and system metadata can be set. Custom
            metadata are keys and values defined by the user. System metadata
            are keys defined by the Object Store and values defined by the
            user. The system metadata keys are:

            - `content_type`
            - `is_content_type_detected`
            - `versions_location`
            - `read_ACL`
            - `write_ACL`
            - `sync_to`
            - `sync_key`
        """
    res = self._get_resource(_container.Container, container)
    res.set_metadata(self, metadata, refresh=refresh)
    return res