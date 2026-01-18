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
def docker_inspect(args: CommonConfig, identifier: str, always: bool=False) -> DockerInspect:
    """
    Return the results of `docker container inspect` for the specified container.
    Raises a ContainerNotFoundError if the container was not found.
    """
    try:
        stdout = docker_command(args, ['container', 'inspect', identifier], capture=True, always=always)[0]
    except SubprocessError as ex:
        stdout = ex.stdout
    if args.explain and (not always):
        items = []
    else:
        items = json.loads(stdout)
    if len(items) == 1:
        return DockerInspect(args, items[0])
    raise ContainerNotFoundError(identifier)