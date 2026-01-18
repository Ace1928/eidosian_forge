import functools
import logging
import os
import shutil
import sys
import uuid
import zipfile
from optparse import Values
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Union
from pip._vendor.packaging.markers import Marker
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import Version
from pip._vendor.packaging.version import parse as parse_version
from pip._vendor.pyproject_hooks import BuildBackendHookCaller
from pip._internal.build_env import BuildEnvironment, NoOpBuildEnvironment
from pip._internal.exceptions import InstallationError, PreviousBuildDirError
from pip._internal.locations import get_scheme
from pip._internal.metadata import (
from pip._internal.metadata.base import FilesystemWheel
from pip._internal.models.direct_url import DirectUrl
from pip._internal.models.link import Link
from pip._internal.operations.build.metadata import generate_metadata
from pip._internal.operations.build.metadata_editable import generate_editable_metadata
from pip._internal.operations.build.metadata_legacy import (
from pip._internal.operations.install.editable_legacy import (
from pip._internal.operations.install.wheel import install_wheel
from pip._internal.pyproject import load_pyproject_toml, make_pyproject_path
from pip._internal.req.req_uninstall import UninstallPathSet
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.misc import (
from pip._internal.utils.packaging import safe_extra
from pip._internal.utils.subprocess import runner_with_spinner_message
from pip._internal.utils.temp_dir import TempDirectory, tempdir_kinds
from pip._internal.utils.unpacking import unpack_file
from pip._internal.utils.virtualenv import running_under_virtualenv
from pip._internal.vcs import vcs
def check_if_exists(self, use_user_site: bool) -> None:
    """Find an installed distribution that satisfies or conflicts
        with this requirement, and set self.satisfied_by or
        self.should_reinstall appropriately.
        """
    if self.req is None:
        return
    existing_dist = get_default_environment().get_distribution(self.req.name)
    if not existing_dist:
        return
    version_compatible = self.req.specifier.contains(existing_dist.version, prereleases=True)
    if not version_compatible:
        self.satisfied_by = None
        if use_user_site:
            if existing_dist.in_usersite:
                self.should_reinstall = True
            elif running_under_virtualenv() and existing_dist.in_site_packages:
                raise InstallationError(f'Will not install to the user site because it will lack sys.path precedence to {existing_dist.raw_name} in {existing_dist.location}')
        else:
            self.should_reinstall = True
    elif self.editable:
        self.should_reinstall = True
        self.satisfied_by = None
    else:
        self.satisfied_by = existing_dist