from __future__ import annotations
import abc
import dataclasses
import json
import os
import re
import stat
import traceback
import uuid
import time
import typing as t
from .http import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .ci import (
from .data import (
def ssh_key_callback(payload_config: PayloadConfig) -> None:
    """
            Add the SSH keys to the payload file list.
            They are either outside the source tree or in the cache dir which is ignored by default.
            """
    files = payload_config.files
    permissions = payload_config.permissions
    files.append((key, os.path.relpath(key_dst, data_context().content.root)))
    files.append((pub, os.path.relpath(pub_dst, data_context().content.root)))
    permissions[os.path.relpath(key_dst, data_context().content.root)] = stat.S_IRUSR | stat.S_IWUSR