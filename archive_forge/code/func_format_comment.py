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
def format_comment(self, template: str, targets: list[str]) -> t.Optional[str]:
    """Format and return a comment based on the given template and targets, or None if there are no targets."""
    if not targets:
        return None
    tests = '\n'.join(('- %s' % target for target in targets))
    data = dict(explain_url=self.EXPLAIN_URL, tests=tests)
    message = textwrap.dedent(template).strip().format(**data)
    return message