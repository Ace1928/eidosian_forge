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
def get_playbook_variables(self) -> dict[str, str]:
    """Return a dictionary of variables for setup and teardown of Windows coverage."""
    return dict(remote_temp_path=self.remote_temp_path)