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
def save_linked_requirement(self, req: InstallRequirement) -> None:
    assert self.download_dir is not None
    assert req.link is not None
    link = req.link
    if link.is_vcs or (link.is_existing_dir() and req.editable):
        req.archive(self.download_dir)
        return
    if link.is_existing_dir():
        logger.debug('Not copying link to destination directory since it is a directory: %s', link)
        return
    if req.local_file_path is None:
        return
    download_location = os.path.join(self.download_dir, link.filename)
    if not os.path.exists(download_location):
        shutil.copy(req.local_file_path, download_location)
        download_path = display_path(download_location)
        logger.info('Saved %s', download_path)