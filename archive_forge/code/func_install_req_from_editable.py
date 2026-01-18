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
def install_req_from_editable(editable_req: str, comes_from: Optional[Union[InstallRequirement, str]]=None, *, use_pep517: Optional[bool]=None, isolated: bool=False, global_options: Optional[List[str]]=None, hash_options: Optional[Dict[str, List[str]]]=None, constraint: bool=False, user_supplied: bool=False, permit_editable_wheels: bool=False, config_settings: Optional[Dict[str, Union[str, List[str]]]]=None) -> InstallRequirement:
    parts = parse_req_from_editable(editable_req)
    return InstallRequirement(parts.requirement, comes_from=comes_from, user_supplied=user_supplied, editable=True, permit_editable_wheels=permit_editable_wheels, link=parts.link, constraint=constraint, use_pep517=use_pep517, isolated=isolated, global_options=global_options, hash_options=hash_options, config_settings=config_settings, extras=parts.extras)