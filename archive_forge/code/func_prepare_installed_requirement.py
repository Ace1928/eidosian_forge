import mimetypes
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.distributions import make_distribution_for_install_requirement
from pip._internal.distributions.installed import InstalledDistribution
from pip._internal.exceptions import (
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution, get_metadata_distribution
from pip._internal.models.direct_url import ArchiveInfo
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.network.download import BatchDownloader, Downloader
from pip._internal.network.lazy_wheel import (
from pip._internal.network.session import PipSession
from pip._internal.operations.build.build_tracker import BuildTracker
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils._log import getLogger
from pip._internal.utils.direct_url_helpers import (
from pip._internal.utils.hashes import Hashes, MissingHashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import (
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.unpacking import unpack_file
from pip._internal.vcs import vcs
def prepare_installed_requirement(self, req: InstallRequirement, skip_reason: str) -> BaseDistribution:
    """Prepare an already-installed requirement."""
    assert req.satisfied_by, "req should have been satisfied but isn't"
    assert skip_reason is not None, f'did not get skip reason skipped but req.satisfied_by is set to {req.satisfied_by}'
    logger.info('Requirement %s: %s (%s)', skip_reason, req, req.satisfied_by.version)
    with indent_log():
        if self.require_hashes:
            logger.debug('Since it is already installed, we are trusting this package without checking its hash. To ensure a completely repeatable environment, install into an empty virtualenv.')
        return InstalledDistribution(req).get_metadata_distribution()