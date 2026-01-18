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
def get_cloud_environment(args: IntegrationConfig, target: IntegrationTarget) -> t.Optional[CloudEnvironment]:
    """Return the cloud environment for the given target, or None if no cloud environment is used for the target."""
    cloud_platform = get_cloud_platform(target)
    if not cloud_platform:
        return None
    return get_environment_plugins()[cloud_platform](args)