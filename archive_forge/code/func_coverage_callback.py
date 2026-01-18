from __future__ import annotations
import collections.abc as c
import os
import json
import typing as t
from ...target import (
from ...io import (
from ...util import (
from ...util_common import (
from ...executor import (
from ...data import (
from ...host_configs import (
from ...provisioning import (
from . import (
def coverage_callback(payload_config: PayloadConfig) -> None:
    """Add the coverage files to the payload file list."""
    display.info('Including %d exported coverage file(s) in payload.' % len(pairs), verbosity=1)
    files = payload_config.files
    files.extend(pairs)