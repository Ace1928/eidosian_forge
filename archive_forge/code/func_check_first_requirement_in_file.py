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
def check_first_requirement_in_file(filename: str) -> None:
    """Check if file is parsable as a requirements file.

    This is heavily based on ``pkg_resources.parse_requirements``, but
    simplified to just check the first meaningful line.

    :raises InvalidRequirement: If the first meaningful line cannot be parsed
        as an requirement.
    """
    with open(filename, encoding='utf-8', errors='ignore') as f:
        lines = (line for line in (line.strip() for line in f) if line and (not line.startswith('#')))
        for line in lines:
            if ' #' in line:
                line = line[:line.find(' #')]
            if line.endswith('\\'):
                line = line[:-2].strip() + next(lines, '')
            Requirement(line)
            return