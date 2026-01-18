import asyncio
import binascii
from collections import defaultdict
import contextlib
import errno
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, unquote, urlparse, parse_qsl, urlunparse
import warnings
from inspect import signature
from pathlib import Path
from subprocess import list2cmdline
from typing import (
import psutil
from google.protobuf import json_format
import ray
import ray._private.ray_constants as ray_constants
from ray.core.generated.runtime_env_common_pb2 import (
def split_address(address: str) -> Tuple[str, str]:
    """Splits address into a module string (scheme) and an inner_address.

    We use a custom splitting function instead of urllib because
    PEP allows "underscores" in a module names, while URL schemes do not
    allow them.

    Args:
        address: The address to split.

    Returns:
        A tuple of (scheme, inner_address).

    Raises:
        ValueError: If the address does not contain '://'.

    Examples:
        >>> split_address("ray://my_cluster")
        ('ray', 'my_cluster')
    """
    if '://' not in address:
        raise ValueError("Address must contain '://'")
    module_string, inner_address = address.split('://', maxsplit=1)
    return (module_string, inner_address)