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
def get_http_url(link: Link, download: Downloader, download_dir: Optional[str]=None, hashes: Optional[Hashes]=None) -> File:
    temp_dir = TempDirectory(kind='unpack', globally_managed=True)
    already_downloaded_path = None
    if download_dir:
        already_downloaded_path = _check_download_dir(link, download_dir, hashes)
    if already_downloaded_path:
        from_path = already_downloaded_path
        content_type = None
    else:
        from_path, content_type = download(link, temp_dir.path)
        if hashes:
            hashes.check_against_path(from_path)
    return File(from_path, content_type)