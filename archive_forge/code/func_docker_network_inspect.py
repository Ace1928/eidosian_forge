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
def docker_network_inspect(args: CommonConfig, network: str, always: bool=False) -> t.Optional[DockerNetworkInspect]:
    """
    Return the results of `docker network inspect` for the specified network or None if the network does not exist.
    """
    try:
        stdout = docker_command(args, ['network', 'inspect', network], capture=True, always=always)[0]
    except SubprocessError:
        stdout = '[]'
    if args.explain and (not always):
        items = []
    else:
        items = json.loads(stdout)
    if len(items) == 1:
        return DockerNetworkInspect(args, items[0])
    return None