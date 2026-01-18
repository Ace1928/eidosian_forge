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
def direct_url(self) -> Optional[DirectUrl]:
    """Obtain a DirectUrl from this distribution.

        Returns None if the distribution has no `direct_url.json` metadata,
        or if `direct_url.json` is invalid.
        """
    try:
        content = self.read_text(DIRECT_URL_METADATA_NAME)
    except FileNotFoundError:
        return None
    try:
        return DirectUrl.from_json(content)
    except (UnicodeDecodeError, json.JSONDecodeError, DirectUrlValidationError) as e:
        logger.warning('Error parsing %s for %s: %s', DIRECT_URL_METADATA_NAME, self.canonical_name, e)
        return None