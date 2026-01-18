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
def get_cloud_platform(target: IntegrationTarget) -> t.Optional[str]:
    """Return the name of the cloud platform used for the given target, or None if no cloud platform is used."""
    cloud_platforms = set((a.split('/')[1] for a in target.aliases if a.startswith('cloud/') and a.endswith('/') and (a != 'cloud/')))
    if not cloud_platforms:
        return None
    if len(cloud_platforms) == 1:
        cloud_platform = cloud_platforms.pop()
        if cloud_platform not in get_provider_plugins():
            raise ApplicationError('Target %s aliases contains unknown cloud platform: %s' % (target.name, cloud_platform))
        return cloud_platform
    raise ApplicationError('Target %s aliases contains multiple cloud platforms: %s' % (target.name, ', '.join(sorted(cloud_platforms))))