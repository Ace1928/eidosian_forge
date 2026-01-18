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
def prepare_linked_requirements_more(self, reqs: Iterable[InstallRequirement], parallel_builds: bool=False) -> None:
    """Prepare linked requirements more, if needed."""
    reqs = [req for req in reqs if req.needs_more_preparation]
    for req in reqs:
        if self.download_dir is not None and req.link.is_wheel:
            hashes = self._get_linked_req_hashes(req)
            file_path = _check_download_dir(req.link, self.download_dir, hashes)
            if file_path is not None:
                self._downloaded[req.link.url] = file_path
                req.needs_more_preparation = False
    partially_downloaded_reqs: List[InstallRequirement] = []
    for req in reqs:
        if req.needs_more_preparation:
            partially_downloaded_reqs.append(req)
        else:
            self._prepare_linked_requirement(req, parallel_builds)
    self._complete_partial_requirements(partially_downloaded_reqs, parallel_builds=parallel_builds)