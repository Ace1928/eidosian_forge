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
def supports_version(adapter: ks_adapter.Adapter, version: str, raise_exception: bool=False) -> bool:
    """Determine if the given adapter supports the given version.

    Checks the version asserted by the service and ensures this matches the
    provided version. ``version`` can be a major version or a major-minor
    version

    :param adapter: :class:`~keystoneauth1.adapter.Adapter` instance.
    :param version: String containing the desired version.
    :param raise_exception: Raise exception when requested version
        is not supported by the server.
    :returns: ``True`` if the service supports the version, else ``False``.
    :raises: :class:`~openstack.exceptions.SDKException` when
        ``raise_exception`` is ``True`` and requested version is not supported.
    """
    required = discover.normalize_version_number(version)
    if discover.version_match(required, adapter.get_api_major_version()):
        return True
    if raise_exception:
        raise exceptions.SDKException(f'Required version {version} is not supported by the server')
    return False