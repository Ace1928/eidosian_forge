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
def setup_target(self) -> None:
    """Perform setup for code coverage on the target."""
    if not self.target_profile:
        return
    if isinstance(self.target_profile, ControllerProfile):
        return
    self.run_playbook('posix_coverage_setup.yml', self.get_playbook_variables())