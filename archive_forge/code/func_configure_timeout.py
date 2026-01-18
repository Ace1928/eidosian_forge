from __future__ import annotations
import dataclasses
import datetime
import functools
import os
import signal
import time
import typing as t
from .io import (
from .config import (
from .util import (
from .thread import (
from .constants import (
from .test import (
def configure_timeout(args: CommonConfig) -> None:
    """Configure the timeout."""
    if isinstance(args, TestConfig):
        configure_test_timeout(args)