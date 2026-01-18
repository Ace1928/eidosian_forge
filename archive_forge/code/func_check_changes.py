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
def check_changes(self, args: SanityConfig, results: Results) -> None:
    """Check changes and store results in the provided result dictionary."""
    integration_targets = list(walk_integration_targets())
    module_targets = list(walk_module_targets())
    integration_targets_by_name = dict(((target.name, target) for target in integration_targets))
    module_names_by_path = dict(((target.path, target.module) for target in module_targets))
    disabled_targets = []
    unstable_targets = []
    unsupported_targets = []
    for command in [command for command in args.metadata.change_description.focused_command_targets if 'integration' in command]:
        for target in args.metadata.change_description.focused_command_targets[command]:
            if self.DISABLED in integration_targets_by_name[target].aliases:
                disabled_targets.append(target)
            elif self.UNSTABLE in integration_targets_by_name[target].aliases:
                unstable_targets.append(target)
            elif self.UNSUPPORTED in integration_targets_by_name[target].aliases:
                unsupported_targets.append(target)
    untested_modules = []
    for path in args.metadata.change_description.no_integration_paths:
        module = module_names_by_path.get(path)
        if module:
            untested_modules.append(module)
    comments = [self.format_comment(self.TEMPLATE_DISABLED, disabled_targets), self.format_comment(self.TEMPLATE_UNSTABLE, unstable_targets), self.format_comment(self.TEMPLATE_UNSUPPORTED, unsupported_targets), self.format_comment(self.TEMPLATE_UNTESTED, untested_modules)]
    comments = [comment for comment in comments if comment]
    labels = dict(needs_tests=bool(untested_modules), disabled_tests=bool(disabled_targets), unstable_tests=bool(unstable_targets), unsupported_tests=bool(unsupported_targets))
    results.comments += comments
    results.labels.update(labels)