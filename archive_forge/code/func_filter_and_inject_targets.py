from __future__ import annotations
import abc
import glob
import hashlib
import json
import os
import pathlib
import re
import collections
import collections.abc as c
import typing as t
from ...constants import (
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...target import (
from ...executor import (
from ...python_requirements import (
from ...config import (
from ...test import (
from ...data import (
from ...content_config import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...venv import (
@staticmethod
def filter_and_inject_targets(test: SanityTest, targets: c.Iterable[TestTarget]) -> list[TestTarget]:
    """Filter and inject targets based on test requirements and the given target list."""
    test_targets = list(targets)
    if not test.include_symlinks:
        test_targets = [target for target in test_targets if not target.symlink]
    if not test.include_directories or not test.include_symlinks:
        test_targets = [target for target in test_targets if not target.path.endswith(os.path.sep)]
    if test.include_directories:
        test_targets += tuple((TestTarget(path, None, None, '') for path in paths_to_dirs([target.path for target in test_targets])))
        if not test.include_symlinks:
            test_targets = [target for target in test_targets if not target.symlink]
    return test_targets