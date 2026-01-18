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
def create_ssh_command(ssh: SshConnectionDetail, options: t.Optional[dict[str, t.Union[str, int]]]=None, cli_args: list[str]=None, command: t.Optional[str]=None) -> list[str]:
    """Create an SSH command using the specified options."""
    cmd = ['ssh', '-n', '-i', ssh.identity_file]
    if not command:
        cmd.append('-N')
    if ssh.port:
        cmd.extend(['-p', str(ssh.port)])
    if ssh.user:
        cmd.extend(['-l', ssh.user])
    ssh_options: dict[str, t.Union[int, str]] = dict(BatchMode='yes', ExitOnForwardFailure='yes', LogLevel='ERROR', ServerAliveCountMax=4, ServerAliveInterval=15, StrictHostKeyChecking='no', UserKnownHostsFile='/dev/null')
    ssh_options.update(options or {})
    cmd.extend(ssh_options_to_list(ssh_options))
    cmd.extend(cli_args or [])
    cmd.append(ssh.host)
    if command:
        cmd.append(command)
    return cmd