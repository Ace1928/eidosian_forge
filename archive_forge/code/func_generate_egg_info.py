from __future__ import annotations
import json
import os
import shutil
import typing as t
from .constants import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .python_requirements import (
from .host_configs import (
from .thread import (
def generate_egg_info(path: str) -> None:
    """Generate an egg-info in the specified base directory."""
    pkg_info = '\nMetadata-Version: 1.0\nName: ansible\nVersion: %s\nPlatform: UNKNOWN\nSummary: Radically simple IT automation\nAuthor-email: info@ansible.com\nLicense: GPLv3+\n' % get_ansible_version()
    pkg_info_path = os.path.join(path, 'ansible_core.egg-info', 'PKG-INFO')
    if os.path.exists(pkg_info_path):
        return
    write_text_file(pkg_info_path, pkg_info.lstrip(), create_directories=True)