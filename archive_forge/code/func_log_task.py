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
@logs_state_cli_group.command(name='task')
@click.option('--id', 'task_id', required=True, type=str, help='Retrieves the logs from the task with this task id.')
@click.option('--attempt-number', '-a', required=False, type=int, default=0, help='Retrieves the logs from the attempt, default to 0')
@address_option
@log_follow_option
@log_interval_option
@log_tail_option
@log_timeout_option
@log_suffix_option
@click.pass_context
@PublicAPI(stability='stable')
def log_task(ctx, task_id: Optional[str], attempt_number: int, address: Optional[str], follow: bool, interval: float, tail: int, timeout: int, err: bool):
    """Get logs associated with a task.

    Example:

        Follow the log file from a task with task id = ABCDEFG

        ```
        ray logs tasks --id ABCDEFG --follow
        ```

        Get the log from a retry attempt 1 from a task.

        ```
        ray logs tasks --id ABCDEFG -a 1
        ```

    Note: If a task is from a concurrent actor (i.e. an async actor or
    a threaded actor), the log of the tasks are expected to be interleaved.
    Please use `ray logs actor --id <actor_id>` for the entire actor log.

    Raises:
        :class:`RayStateApiException <ray.util.state.exception.RayStateApiException>`
            if the CLI is failed to query the data.
        MissingParameter if inputs are missing.
    """
    _print_log(address=address, task_id=task_id, attempt_number=attempt_number, follow=follow, tail=tail, interval=interval, timeout=timeout, suffix='err' if err else 'out')