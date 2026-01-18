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
@staticmethod
def get_relative_in_tree_private_key_path() -> str:
    """Return the ansible-test SSH private key path relative to the content tree."""
    temp_dir = ResultType.TMP.relative_path
    key = os.path.join(temp_dir, SshKey.KEY_NAME)
    return key