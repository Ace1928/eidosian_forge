from __future__ import annotations
import dataclasses
import json
import textwrap
import os
import re
import typing as t
from . import (
from ...test import (
from ...config import (
from ...target import (
from ..integration.cloud import (
from ...io import (
from ...util import (
from ...util_common import (
from ...host_configs import (
def check_windows_targets(self) -> list[SanityMessage]:
    """Check Windows integration test targets and return messages with any issues found."""
    windows_targets = tuple(walk_windows_integration_targets())
    messages = []
    messages += self.check_ci_group(targets=windows_targets, find=self.format_test_group_alias('windows'), find_incidental=['%s/windows/incidental/' % self.TEST_ALIAS_PREFIX])
    return messages