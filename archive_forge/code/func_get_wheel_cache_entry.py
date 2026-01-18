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
def get_wheel_cache_entry(self, link: Link, name: Optional[str]) -> Optional[CacheEntry]:
    """Look up the link in the wheel cache.

        If ``preparer.require_hashes`` is True, don't use the wheel cache,
        because cached wheels, always built locally, have different hashes
        than the files downloaded from the index server and thus throw false
        hash mismatches. Furthermore, cached wheels at present have
        nondeterministic contents due to file modification times.
        """
    if self._wheel_cache is None:
        return None
    return self._wheel_cache.get_cache_entry(link=link, package_name=name, supported_tags=get_supported())