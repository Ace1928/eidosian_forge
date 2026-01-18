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
def install_req_extend_extras(ireq: InstallRequirement, extras: Collection[str]) -> InstallRequirement:
    """
    Returns a copy of an installation requirement with some additional extras.
    Makes a shallow copy of the ireq object.
    """
    result = copy.copy(ireq)
    result.extras = {*ireq.extras, *extras}
    result.req = _set_requirement_extras(ireq.req, result.extras) if ireq.req is not None else None
    return result