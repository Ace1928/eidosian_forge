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
def iter_installed_distributions(self, local_only: bool=True, skip: Container[str]=stdlib_pkgs, include_editables: bool=True, editables_only: bool=False, user_only: bool=False) -> Iterator[BaseDistribution]:
    """Return a list of installed distributions.

        This is based on ``iter_all_distributions()`` with additional filtering
        options. Note that ``iter_installed_distributions()`` without arguments
        is *not* equal to ``iter_all_distributions()``, since some of the
        configurations exclude packages by default.

        :param local_only: If True (default), only return installations
        local to the current virtualenv, if in a virtualenv.
        :param skip: An iterable of canonicalized project names to ignore;
            defaults to ``stdlib_pkgs``.
        :param include_editables: If False, don't report editables.
        :param editables_only: If True, only report editables.
        :param user_only: If True, only report installations in the user
        site directory.
        """
    it = self.iter_all_distributions()
    if local_only:
        it = (d for d in it if d.local)
    if not include_editables:
        it = (d for d in it if not d.editable)
    if editables_only:
        it = (d for d in it if d.editable)
    if user_only:
        it = (d for d in it if d.in_usersite)
    return (d for d in it if d.canonical_name not in skip)