from __future__ import annotations
import os
import urllib.parse
from .io import (
from .config import (
from .host_configs import (
from .util import (
from .util_common import (
from .docker_util import (
from .containers import (
from .ansible_util import (
from .host_profiles import (
from .inventory import (
def configure_controller_pypi_proxy(args: EnvironmentConfig, profile: HostProfile, pypi_endpoint: str, pypi_hostname: str) -> None:
    """Configure the controller environment to use a PyPI proxy."""
    configure_pypi_proxy_pip(args, profile, pypi_endpoint, pypi_hostname)
    configure_pypi_proxy_easy_install(args, profile, pypi_endpoint)