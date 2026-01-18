import logging
import sys
from typing import TYPE_CHECKING, Any, FrozenSet, Iterable, Optional, Tuple, Union, cast
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import Version
from pip._internal.exceptions import (
from pip._internal.metadata import BaseDistribution
from pip._internal.models.link import Link, links_equivalent
from pip._internal.models.wheel import Wheel
from pip._internal.req.constructors import (
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.direct_url_helpers import direct_url_from_link
from pip._internal.utils.misc import normalize_version_info
from .base import Candidate, CandidateVersion, Requirement, format_name
def make_install_req_from_link(link: Link, template: InstallRequirement) -> InstallRequirement:
    assert not template.editable, 'template is editable'
    if template.req:
        line = str(template.req)
    else:
        line = link.url
    ireq = install_req_from_line(line, user_supplied=template.user_supplied, comes_from=template.comes_from, use_pep517=template.use_pep517, isolated=template.isolated, constraint=template.constraint, global_options=template.global_options, hash_options=template.hash_options, config_settings=template.config_settings)
    ireq.original_link = template.original_link
    ireq.link = link
    ireq.extras = template.extras
    return ireq