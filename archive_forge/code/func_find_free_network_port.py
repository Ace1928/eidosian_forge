import os
import socket
from typing_extensions import override
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.utilities.rank_zero import rank_zero_only
def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.

    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port