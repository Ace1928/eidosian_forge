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
def get_docker_details(args: EnvConfig) -> dict[str, t.Any]:
    """Return details about docker."""
    docker = get_docker_command()
    executable = None
    info = None
    version = None
    if docker:
        executable = docker.executable
        try:
            docker_info = get_docker_info(args)
        except ApplicationError as ex:
            display.warning(str(ex))
        else:
            info = docker_info.info
            version = docker_info.version
    docker_details = dict(executable=executable, info=info, version=version)
    return docker_details