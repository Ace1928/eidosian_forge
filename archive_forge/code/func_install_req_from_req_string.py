import copy
import logging
import os
import re
from typing import Collection, Dict, List, Optional, Set, Tuple, Union
from pip._vendor.packaging.markers import Marker
from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
from pip._vendor.packaging.specifiers import Specifier
from pip._internal.exceptions import InstallationError
from pip._internal.models.index import PyPI, TestPyPI
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.req.req_file import ParsedRequirement
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import is_installable_dir
from pip._internal.utils.packaging import get_requirement
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs import is_url, vcs
def install_req_from_req_string(req_string: str, comes_from: Optional[InstallRequirement]=None, isolated: bool=False, use_pep517: Optional[bool]=None, user_supplied: bool=False) -> InstallRequirement:
    try:
        req = get_requirement(req_string)
    except InvalidRequirement:
        raise InstallationError(f"Invalid requirement: '{req_string}'")
    domains_not_allowed = [PyPI.file_storage_domain, TestPyPI.file_storage_domain]
    if req.url and comes_from and comes_from.link and (comes_from.link.netloc in domains_not_allowed):
        raise InstallationError(f'Packages installed from PyPI cannot depend on packages which are not also hosted on PyPI.\n{comes_from.name} depends on {req} ')
    return InstallRequirement(req, comes_from, isolated=isolated, use_pep517=use_pep517, user_supplied=user_supplied)