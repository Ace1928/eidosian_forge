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
def check_ci_group(self, targets: tuple[CompletionTarget, ...], find: str, find_incidental: t.Optional[list[str]]=None) -> list[SanityMessage]:
    """Check the CI groups set in the provided targets and return a list of messages with any issues found."""
    all_paths = set((target.path for target in targets))
    supported_paths = set((target.path for target in filter_targets(targets, [find], errors=False)))
    unsupported_paths = set((target.path for target in filter_targets(targets, [self.UNSUPPORTED], errors=False)))
    if find_incidental:
        incidental_paths = set((target.path for target in filter_targets(targets, find_incidental, errors=False)))
    else:
        incidental_paths = set()
    unassigned_paths = all_paths - supported_paths - unsupported_paths - incidental_paths
    conflicting_paths = supported_paths & unsupported_paths
    unassigned_message = 'missing alias `%s` or `%s`' % (find.strip('/'), self.UNSUPPORTED.strip('/'))
    conflicting_message = 'conflicting alias `%s` and `%s`' % (find.strip('/'), self.UNSUPPORTED.strip('/'))
    messages = []
    for path in unassigned_paths:
        if path == 'test/integration/targets/ansible-test-container':
            continue
        messages.append(SanityMessage(unassigned_message, '%s/aliases' % path))
    for path in conflicting_paths:
        messages.append(SanityMessage(conflicting_message, '%s/aliases' % path))
    return messages