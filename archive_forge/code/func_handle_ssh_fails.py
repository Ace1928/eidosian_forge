import os
import re
import subprocess
import sys
import tempfile
import time
from ray.autoscaler._private.cli_logger import cf, cli_logger
def handle_ssh_fails(e, first_conn_refused_time, retry_interval):
    """Handle SSH system failures coming from a subprocess.

    Args:
        e: The `ProcessRunnerException` to handle.
        first_conn_refused_time:
            The time (as reported by this function) or None,
            indicating the last time a CONN_REFUSED error was caught.

            After exceeding a patience value, the program will be aborted
            since SSH will likely never recover.
        retry_interval: The interval after which the command will be retried,
                        used here just to inform the user.
    """
    if e.msg_type != 'ssh_command_failed':
        return
    if e.special_case == 'ssh_conn_refused':
        if first_conn_refused_time is not None and time.time() - first_conn_refused_time > CONN_REFUSED_PATIENCE:
            cli_logger.error('SSH connection was being refused for {} seconds. Head node assumed unreachable.', cf.bold(str(CONN_REFUSED_PATIENCE)))
            cli_logger.abort("Check the node's firewall settings and the cloud network configuration.")
        cli_logger.warning('SSH connection was refused.')
        cli_logger.warning('This might mean that the SSH daemon is still setting up, or that the host is inaccessable (e.g. due to a firewall).')
        return time.time()
    if e.special_case in ['ssh_timeout', 'ssh_conn_refused']:
        cli_logger.print('SSH still not available, retrying in {} seconds.', cf.bold(str(retry_interval)))
    else:
        raise e
    return first_conn_refused_time