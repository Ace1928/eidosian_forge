from __future__ import annotations
import datetime
import os
import platform
import sys
import typing as t
from ...config import (
from ...io import (
from ...util import (
from ...util_common import (
from ...docker_util import (
from ...constants import (
from ...ci import (
from ...timeout import (
def list_files_env(args: EnvConfig) -> None:
    """List files on stdout."""
    if not args.list_files:
        return
    for path in data_context().content.all_files():
        display.info(path)