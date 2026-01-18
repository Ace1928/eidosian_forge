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
def docker_pull(args: CommonConfig, image: str) -> None:
    """
    Pull the specified image if it is not available.
    Images without a tag or digest will not be pulled.
    Retries up to 10 times if the pull fails.
    A warning will be shown for any image with volumes defined.
    Images will be pulled only once.
    Concurrent pulls for the same image will block until the first completes.
    """
    with named_lock(f'docker_pull:{image}') as first:
        if first:
            __docker_pull(args, image)