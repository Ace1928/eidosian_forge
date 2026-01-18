import contextlib
import functools
import logging
from typing import (
from pip._vendor.packaging.requirements import InvalidRequirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.resolvelib import ResolutionImpossible
from pip._internal.cache import CacheEntry, WheelCache
from pip._internal.exceptions import (
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution, get_default_environment
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.constructors import (
from pip._internal.req.req_install import (
from pip._internal.resolution.base import InstallRequirementProvider
from pip._internal.utils.compatibility_tags import get_supported
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.packaging import get_requirement
from pip._internal.utils.virtualenv import running_under_virtualenv
from .base import Candidate, CandidateVersion, Constraint, Requirement
from .candidates import (
from .found_candidates import FoundCandidates, IndexCandidateInfo
from .requirements import (
def get_dist_to_uninstall(self, candidate: Candidate) -> Optional[BaseDistribution]:
    dist = self._installed_dists.get(candidate.project_name)
    if dist is None:
        return None
    if not self._use_user_site:
        return dist
    if dist.in_usersite:
        return dist
    if running_under_virtualenv() and dist.in_site_packages:
        message = f'Will not install to the user site because it will lack sys.path precedence to {dist.raw_name} in {dist.location}'
        raise InstallationError(message)
    return None