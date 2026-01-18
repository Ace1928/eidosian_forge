import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from ray.autoscaler._private import commands
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.event_system import CreateClusterEvent  # noqa: F401
from ray.autoscaler._private.event_system import global_event_system  # noqa: F401
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def run_on_cluster(cluster_config: Union[dict, str], *, cmd: Optional[str]=None, run_env: str='auto', tmux: bool=False, stop: bool=False, no_config_cache: bool=False, port_forward: Optional[commands.Port_forward]=None, with_output: bool=False) -> Optional[str]:
    """Runs a command on the specified cluster.

    Args:
        cluster_config (Union[str, dict]): Either the config dict of the
            cluster, or a path pointing to a file containing the config.
        cmd: the command to run, or None for a no-op command.
        run_env: whether to run the command on the host or in a
            container. Select between "auto", "host" and "docker".
        tmux: whether to run in a tmux session
        stop: whether to stop the cluster after command run
        no_config_cache: Whether to disable the config cache and fully
            resolve all environment settings from the Cloud provider again.
        port_forward ( (int,int) or list[(int,int)]): port(s) to forward.
        with_output: Whether to capture command output.

    Returns:
        The output of the command as a string.
    """
    with _as_config_file(cluster_config) as config_file:
        return commands.exec_cluster(config_file, cmd=cmd, run_env=run_env, screen=False, tmux=tmux, stop=stop, start=False, override_cluster_name=None, no_config_cache=no_config_cache, port_forward=port_forward, with_output=with_output)