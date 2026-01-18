from collections.abc import Mapping
import hashlib
import queue
import string
import threading
import time
import typing as ty
import keystoneauth1
from keystoneauth1 import adapter as ks_adapter
from keystoneauth1 import discover
from openstack import _log
from openstack import exceptions
def require_microversion(adapter, required):
    """Require microversion.

    :param adapter: :class:`~keystoneauth1.adapter.Adapter` instance.
    :param str microversion: String containing the desired microversion.
    :raises: :class:`~openstack.exceptions.SDKException` when requested
        microversion is not supported
    """
    supports_microversion(adapter, required, raise_exception=True)