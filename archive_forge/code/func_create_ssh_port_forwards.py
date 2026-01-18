from __future__ import annotations
import dataclasses
import itertools
import json
import os
import random
import re
import subprocess
import shlex
import typing as t
from .encoding import (
from .util import (
from .config import (
def create_ssh_port_forwards(args: EnvironmentConfig, ssh: SshConnectionDetail, forwards: list[tuple[str, int]]) -> SshProcess:
    """
    Create SSH port forwards using the provided list of tuples (target_host, target_port).
    Port bindings will be automatically assigned by SSH and must be collected with a subsequent call to collect_port_forwards.
    """
    options: dict[str, t.Union[str, int]] = dict(LogLevel='INFO')
    cli_args = []
    for forward_host, forward_port in forwards:
        cli_args.extend(['-R', ':'.join([str(0), forward_host, str(forward_port)])])
    process = run_ssh_command(args, ssh, options, cli_args)
    process.pending_forwards = forwards
    return process