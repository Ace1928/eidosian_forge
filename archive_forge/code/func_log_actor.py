import json
import logging
from datetime import datetime
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple
import click
import yaml
import ray._private.services as services
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.state import (
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException
from ray.util.annotations import PublicAPI
@logs_state_cli_group.command(name='actor')
@click.option('--id', '-a', required=False, type=str, default=None, help='Retrieves the logs corresponding to this ActorID.')
@click.option('--pid', '-pid', required=False, type=str, default=None, help='Retrieves the logs from the actor with this pid.')
@address_option
@log_node_id_option
@log_node_ip_option
@log_follow_option
@log_tail_option
@log_interval_option
@log_timeout_option
@log_suffix_option
@click.pass_context
@PublicAPI(stability='stable')
def log_actor(ctx, id: Optional[str], pid: Optional[str], address: Optional[str], node_id: Optional[str], node_ip: Optional[str], follow: bool, tail: int, interval: float, timeout: int, err: bool):
    """Get/List logs associated with an actor.

    Example:

        Follow the log file with an actor id ABCDEFG.

        ```
        ray logs actor --id ABCDEFG --follow
        ```

        Get the actor log from pid 123, ip x.x.x.x
        Note that this goes well with the driver log of Ray which prints
        (ip=x.x.x.x, pid=123, class_name) logs.

        ```
        ray logs actor --pid=123  —ip=x.x.x.x
        ```

        Get the actor err log file.

        ```
        ray logs actor --id ABCDEFG --err
        ```

    Raises:
        :class:`RayStateApiException <ray.util.state.exception.RayStateApiException>`
            if the CLI is failed to query the data.
        MissingParameter if inputs are missing.
    """
    if pid is None and id is None:
        raise click.MissingParameter(message='At least one of `--pid` and `--id` has to be set', param_type='option')
    _print_log(address=address, node_id=node_id, node_ip=node_ip, pid=pid, actor_id=id, tail=tail, follow=follow, interval=interval, timeout=timeout, suffix='err' if err else 'out')