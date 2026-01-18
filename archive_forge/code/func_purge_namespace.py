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
@staticmethod
def purge_namespace(namespace: t.Union[argparse.Namespace, types.SimpleNamespace]) -> None:
    """Purge legacy host options fields from the given namespace."""
    for field in dataclasses.fields(LegacyHostOptions):
        if hasattr(namespace, field.name):
            delattr(namespace, field.name)