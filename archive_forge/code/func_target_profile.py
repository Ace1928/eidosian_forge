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
@property
def target_profile(self) -> t.Optional[PosixProfile]:
    """The POSIX target profile, if it uses a different Python interpreter than the controller, otherwise None."""
    return t.cast(PosixProfile, self.profiles[0]) if self.profiles else None