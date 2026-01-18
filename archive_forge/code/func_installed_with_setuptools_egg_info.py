import csv
import email.message
import functools
import json
import logging
import pathlib
import re
import zipfile
from typing import (
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import InvalidSpecifier, SpecifierSet
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import LegacyVersion, Version
from pip._internal.exceptions import NoneMetadataError
from pip._internal.locations import site_packages, user_site
from pip._internal.models.direct_url import (
from pip._internal.utils.compat import stdlib_pkgs  # TODO: Move definition here.
from pip._internal.utils.egg_link import egg_link_path_from_sys_path
from pip._internal.utils.misc import is_local, normalize_path
from pip._internal.utils.urls import url_to_path
from ._json import msg_to_json
@property
def installed_with_setuptools_egg_info(self) -> bool:
    """Whether this distribution is installed with the ``.egg-info`` format.

        This usually indicates the distribution was installed with setuptools
        with an old pip version or with ``single-version-externally-managed``.

        Note that this ensure the metadata store is a directory. distutils can
        also installs an ``.egg-info``, but as a file, not a directory. This
        property is *False* for that case. Also see ``installed_by_distutils``.
        """
    info_location = self.info_location
    if not info_location:
        return False
    if not info_location.endswith('.egg-info'):
        return False
    return pathlib.Path(info_location).is_dir()