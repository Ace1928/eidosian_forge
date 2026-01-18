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
def prepare_editable_requirement(self, req: InstallRequirement) -> BaseDistribution:
    """Prepare an editable requirement."""
    assert req.editable, 'cannot prepare a non-editable req as editable'
    logger.info('Obtaining %s', req)
    with indent_log():
        if self.require_hashes:
            raise InstallationError(f'The editable requirement {req} cannot be installed when requiring hashes, because there is no single file to hash.')
        req.ensure_has_source_dir(self.src_dir)
        req.update_editable()
        assert req.source_dir
        req.download_info = direct_url_for_editable(req.unpacked_source_directory)
        dist = _get_prepared_distribution(req, self.build_tracker, self.finder, self.build_isolation, self.check_build_deps)
        req.check_if_exists(self.use_user_site)
    return dist