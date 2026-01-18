from __future__ import annotations
import abc
import datetime
import os
import re
import tempfile
import time
import typing as t
from ....encoding import (
from ....io import (
from ....util import (
from ....util_common import (
from ....target import (
from ....config import (
from ....ci import (
from ....data import (
from ....docker_util import (
def get_cloud_platforms(args: TestConfig, targets: t.Optional[tuple[IntegrationTarget, ...]]=None) -> list[str]:
    """Return cloud platform names for the specified targets."""
    if isinstance(args, IntegrationConfig):
        if args.list_targets:
            return []
    if targets is None:
        cloud_platforms = set(args.metadata.cloud_config or [])
    else:
        cloud_platforms = set((get_cloud_platform(target) for target in targets))
    cloud_platforms.discard(None)
    return sorted(cloud_platforms)