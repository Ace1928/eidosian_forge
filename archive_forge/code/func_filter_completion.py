from __future__ import annotations
import abc
import dataclasses
import enum
import os
import typing as t
from .constants import (
from .util import (
from .data import (
from .become import (
def filter_completion(completion: dict[str, TCompletionConfig], controller_only: bool=False, include_defaults: bool=False) -> dict[str, TCompletionConfig]:
    """Return the given completion dictionary, filtering out configs which do not support the controller if controller_only is specified."""
    if controller_only:
        completion = {name: t.cast(TCompletionConfig, config) for name, config in completion.items() if isinstance(config, PosixCompletionConfig) and config.controller_supported}
    if not include_defaults:
        completion = {name: config for name, config in completion.items() if not config.is_default}
    return completion