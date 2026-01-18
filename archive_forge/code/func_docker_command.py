from __future__ import annotations
import dataclasses
import enum
import json
import os
import pathlib
import re
import socket
import time
import urllib.parse
import typing as t
from .util import (
from .util_common import (
from .config import (
from .thread import (
from .cgroup import (
def docker_command(args: CommonConfig, cmd: list[str], capture: bool, stdin: t.Optional[t.IO[bytes]]=None, stdout: t.Optional[t.IO[bytes]]=None, interactive: bool=False, output_stream: t.Optional[OutputStream]=None, always: bool=False, data: t.Optional[str]=None) -> tuple[t.Optional[str], t.Optional[str]]:
    """Run the specified docker command."""
    env = docker_environment()
    command = [require_docker().command]
    if command[0] == 'podman' and get_podman_remote():
        command.append('--remote')
    return run_command(args, command + cmd, env=env, capture=capture, stdin=stdin, stdout=stdout, interactive=interactive, always=always, output_stream=output_stream, data=data)