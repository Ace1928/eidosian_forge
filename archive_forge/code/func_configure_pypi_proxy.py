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
def configure_pypi_proxy(args: EnvironmentConfig, profile: HostProfile) -> None:
    """Configure the environment to use a PyPI proxy, if present."""
    if args.pypi_endpoint:
        pypi_endpoint = args.pypi_endpoint
    else:
        containers = get_container_database(args)
        context = containers.data.get(HostType.control if profile.controller else HostType.managed, {}).get('__pypi_proxy__')
        if not context:
            return
        access = list(context.values())[0]
        host = access.host_ip
        port = dict(access.port_map())[3141]
        pypi_endpoint = f'http://{host}:{port}/root/pypi/+simple/'
    pypi_hostname = urllib.parse.urlparse(pypi_endpoint)[1].split(':')[0]
    if profile.controller:
        configure_controller_pypi_proxy(args, profile, pypi_endpoint, pypi_hostname)
    else:
        configure_target_pypi_proxy(args, profile, pypi_endpoint, pypi_hostname)