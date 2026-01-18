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
def requires_python(self) -> SpecifierSet:
    """Value of "Requires-Python:" in distribution metadata.

        If the key does not exist or contains an invalid value, an empty
        SpecifierSet should be returned.
        """
    value = self.metadata.get('Requires-Python')
    if value is None:
        return SpecifierSet()
    try:
        spec = SpecifierSet(str(value))
    except InvalidSpecifier as e:
        message = 'Package %r has an invalid Requires-Python: %s'
        logger.warning(message, self.raw_name, e)
        return SpecifierSet()
    return spec