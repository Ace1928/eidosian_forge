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
def cloud_filter(args: IntegrationConfig, targets: tuple[IntegrationTarget, ...]) -> list[str]:
    """Return a list of target names to exclude based on the given targets."""
    if args.metadata.cloud_config is not None:
        return []
    exclude: list[str] = []
    for provider in get_cloud_providers(args, targets):
        provider.filter(targets, exclude)
    return exclude