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
def controller_targets(mode: TargetMode, options: LegacyHostOptions, controller: ControllerHostConfig) -> list[HostConfig]:
    """Return the configuration for controller targets."""
    python = native_python(options)
    targets: list[HostConfig]
    if python:
        targets = [ControllerConfig(python=python)]
    else:
        targets = default_targets(mode, controller)
    return targets