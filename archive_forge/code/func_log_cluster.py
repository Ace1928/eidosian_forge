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
@logs_state_cli_group.command(name='cluster')
@click.argument('glob_filter', required=False, default='*')
@address_option
@log_node_id_option
@log_node_ip_option
@log_follow_option
@log_tail_option
@log_interval_option
@log_timeout_option
@log_encoding_option
@log_encoding_errors_option
@click.pass_context
@PublicAPI(stability='stable')
def log_cluster(ctx, glob_filter: str, address: Optional[str], node_id: Optional[str], node_ip: Optional[str], follow: bool, tail: int, interval: float, timeout: int, encoding: str, encoding_errors: str):
    """Get/List logs that matches the GLOB_FILTER in the cluster.
    By default, it prints a list of log files that match the filter.
    By default, it prints the head node logs.
    If there's only 1 match, it will print the log file.

    Example:

        Print the last 500 lines of raylet.out on a head node.

        ```
        ray logs [cluster] raylet.out --tail 500
        ```

        Print the last 500 lines of raylet.out on a worker node id A.

        ```
        ray logs [cluster] raylet.out --tail 500 —-node-id A
        ```

        Download the gcs_server.txt file to the local machine.

        ```
        ray logs [cluster] gcs_server.out --tail -1 > gcs_server.txt
        ```

        Follow the log files from the last 100 lines.

        ```
        ray logs [cluster] raylet.out --tail 100 -f
        ```

    Raises:
        :class:`RayStateApiException <ray.util.state.exception.RayStateApiException>` if the CLI
            is failed to query the data.
    """
    if node_id is None and node_ip is None:
        node_ip = _get_head_node_ip(address)
    logs = list_logs(address=address, node_id=node_id, node_ip=node_ip, glob_filter=glob_filter, timeout=timeout)
    log_files_found = []
    for _, log_files in logs.items():
        for log_file in log_files:
            log_files_found.append(log_file)
    if len(log_files_found) != 1:
        if node_id:
            print(f'Node ID: {node_id}')
        elif node_ip:
            print(f'Node IP: {node_ip}')
        print(output_with_format(logs, schema=None, format=AvailableFormat.YAML))
        return
    filename = log_files_found[0]
    _print_log(address=address, node_id=node_id, node_ip=node_ip, filename=filename, tail=tail, follow=follow, interval=interval, timeout=timeout, encoding=encoding, encoding_errors=encoding_errors)