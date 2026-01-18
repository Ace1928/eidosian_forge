from __future__ import annotations
import collections.abc as c
import contextlib
import json
import random
import time
import uuid
import threading
import typing as t
from .util import (
from .util_common import (
from .config import (
from .docker_util import (
from .ansible_util import (
from .core_ci import (
from .target import (
from .ssh import (
from .host_configs import (
from .connections import (
from .thread import (
def get_container_ip_address(args: EnvironmentConfig, container: DockerInspect) -> t.Optional[str]:
    """Return the IP address of the container for the preferred docker network."""
    if container.networks:
        network_name = get_docker_preferred_network_name(args)
        if not network_name:
            network_name = sorted(container.networks.keys()).pop(0)
        ipaddress = container.networks[network_name]['IPAddress']
    else:
        ipaddress = container.network_settings['IPAddress']
    if not ipaddress:
        return None
    return ipaddress