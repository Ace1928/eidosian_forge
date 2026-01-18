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
def get_cloud_providers(args: IntegrationConfig, targets: t.Optional[tuple[IntegrationTarget, ...]]=None) -> list[CloudProvider]:
    """Return a list of cloud providers for the given targets."""
    return [get_provider_plugins()[p](args) for p in get_cloud_platforms(args, targets)]