from __future__ import annotations
import argparse
import collections.abc as c
import dataclasses
import enum
import os
import types
import typing as t
from ..constants import (
from ..util import (
from ..docker_util import (
from ..completion import (
from ..host_configs import (
from ..data import (
def get_fallback_remote_controller() -> str:
    """Return the remote fallback platform for the controller."""
    platform = 'freebsd'
    candidates = [item for item in filter_completion(remote_completion()).values() if item.controller_supported and item.platform == platform]
    fallback = sorted(candidates, key=lambda value: str_to_version(value.version), reverse=True)[0]
    return fallback.name