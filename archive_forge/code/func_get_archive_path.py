from __future__ import annotations
import collections
import contextlib
import json
import os
import tarfile
import typing as t
from . import (
from ...io import (
from ...test import (
from ...target import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...config import (
from ...ci import (
from ...data import (
from ...host_configs import (
from ...git import (
from ...provider.source import (
@staticmethod
def get_archive_path(args: SanityConfig) -> str:
    """Return the path to the original plugin content archive."""
    return os.path.join(ResultType.TMP.path, f'validate-modules-{args.metadata.session_id}.tar')