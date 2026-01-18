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
@cache
def get_cloud_plugins() -> tuple[dict[str, t.Type[CloudProvider]], dict[str, t.Type[CloudEnvironment]]]:
    """Import cloud plugins and load them into the plugin dictionaries."""
    import_plugins('commands/integration/cloud')
    providers: dict[str, t.Type[CloudProvider]] = {}
    environments: dict[str, t.Type[CloudEnvironment]] = {}
    load_plugins(CloudProvider, providers)
    load_plugins(CloudEnvironment, environments)
    return (providers, environments)