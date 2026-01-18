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
def purge_args(args: list[str]) -> list[str]:
    """Purge legacy host options from the given command line arguments."""
    fields: tuple[dataclasses.Field, ...] = dataclasses.fields(LegacyHostOptions)
    filters: dict[str, int] = {get_option_name(field.name): 0 if field.type is t.Optional[bool] else 1 for field in fields}
    return filter_args(args, filters)