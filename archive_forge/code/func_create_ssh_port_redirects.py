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
def create_ssh_port_redirects(args: EnvironmentConfig, ssh: SshConnectionDetail, redirects: list[tuple[int, str, int]]) -> SshProcess:
    """Create SSH port redirections using the provided list of tuples (bind_port, target_host, target_port)."""
    options: dict[str, t.Union[str, int]] = {}
    cli_args = []
    for bind_port, target_host, target_port in redirects:
        cli_args.extend(['-R', ':'.join([str(bind_port), target_host, str(target_port)])])
    process = run_ssh_command(args, ssh, options, cli_args)
    return process