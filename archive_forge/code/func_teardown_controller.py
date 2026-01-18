from __future__ import annotations
import abc
import os
import shutil
import tempfile
import typing as t
import zipfile
from ...io import (
from ...ansible_util import (
from ...config import (
from ...util import (
from ...util_common import (
from ...coverage_util import (
from ...host_configs import (
from ...data import (
from ...host_profiles import (
from ...provisioning import (
from ...connections import (
from ...inventory import (
def teardown_controller(self) -> None:
    """Perform teardown for code coverage on the controller."""
    coverage_temp_path = os.path.join(self.common_temp_path, ResultType.COVERAGE.name)
    platform = get_coverage_platform(self.args.controller)
    for filename in os.listdir(coverage_temp_path):
        shutil.copyfile(os.path.join(coverage_temp_path, filename), os.path.join(ResultType.COVERAGE.path, update_coverage_filename(filename, platform)))
    remove_tree(self.common_temp_path)