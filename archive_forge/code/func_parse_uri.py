import asyncio
import hashlib
import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse
from zipfile import ZipFile
from filelock import FileLock
from ray.util.annotations import DeveloperAPI
from ray._private.ray_constants import (
from ray._private.runtime_env.conda_utils import exec_cmd_stream_to_logger
from ray._private.thirdparty.pathspec import PathSpec
from ray.experimental.internal_kv import (
def parse_uri(pkg_uri: str) -> Tuple[Protocol, str]:
    """
    Parse package uri into protocol and package name based on its format.
    Note that the output of this function is not for handling actual IO, it's
    only for setting up local directory folders by using package name as path.

    >>> parse_uri("https://test.com/file.zip")
    (<Protocol.HTTPS: 'https'>, 'https_test_com_file.zip')

    >>> parse_uri("https://test.com/file.whl")
    (<Protocol.HTTPS: 'https'>, 'file.whl')

    """
    uri = urlparse(pkg_uri)
    try:
        protocol = Protocol(uri.scheme)
    except ValueError as e:
        raise ValueError(f'Invalid protocol for runtime_env URI "{pkg_uri}". Supported protocols: {Protocol._member_names_}. Original error: {e}')
    if protocol in Protocol.remote_protocols():
        if pkg_uri.endswith('.whl'):
            package_name = pkg_uri.split('/')[-1]
        else:
            package_name = f'{protocol.value}_{uri.netloc}{uri.path}'
            disallowed_chars = ['/', ':', '@', '+']
            for disallowed_char in disallowed_chars:
                package_name = package_name.replace(disallowed_char, '_')
            package_name = package_name.replace('.', '_', package_name.count('.') - 1)
    else:
        package_name = uri.netloc
    return (protocol, package_name)