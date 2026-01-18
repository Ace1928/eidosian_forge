import enum
import functools
import itertools
import logging
import re
from typing import TYPE_CHECKING, FrozenSet, Iterable, List, Optional, Set, Tuple, Union
from pip._vendor.packaging import specifiers
from pip._vendor.packaging.tags import Tag
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import _BaseVersion
from pip._vendor.packaging.version import parse as parse_version
from pip._internal.exceptions import (
from pip._internal.index.collector import LinkCollector, parse_links
from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.format_control import FormatControl
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.models.target_python import TargetPython
from pip._internal.models.wheel import Wheel
from pip._internal.req import InstallRequirement
from pip._internal.utils._log import getLogger
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import build_netloc
from pip._internal.utils.packaging import check_requires_python
from pip._internal.utils.unpacking import SUPPORTED_EXTENSIONS
def find_requirement(self, req: InstallRequirement, upgrade: bool) -> Optional[InstallationCandidate]:
    """Try to find a Link matching req

        Expects req, an InstallRequirement and upgrade, a boolean
        Returns a InstallationCandidate if found,
        Raises DistributionNotFound or BestVersionAlreadyInstalled otherwise
        """
    hashes = req.hashes(trust_internet=False)
    best_candidate_result = self.find_best_candidate(req.name, specifier=req.specifier, hashes=hashes)
    best_candidate = best_candidate_result.best_candidate
    installed_version: Optional[_BaseVersion] = None
    if req.satisfied_by is not None:
        installed_version = req.satisfied_by.version

    def _format_versions(cand_iter: Iterable[InstallationCandidate]) -> str:
        return ', '.join(sorted({str(c.version) for c in cand_iter}, key=parse_version)) or 'none'
    if installed_version is None and best_candidate is None:
        logger.critical('Could not find a version that satisfies the requirement %s (from versions: %s)', req, _format_versions(best_candidate_result.iter_all()))
        raise DistributionNotFound(f'No matching distribution found for {req}')

    def _should_install_candidate(candidate: Optional[InstallationCandidate]) -> 'TypeGuard[InstallationCandidate]':
        if installed_version is None:
            return True
        if best_candidate is None:
            return False
        return best_candidate.version > installed_version
    if not upgrade and installed_version is not None:
        if _should_install_candidate(best_candidate):
            logger.debug('Existing installed version (%s) satisfies requirement (most up-to-date version is %s)', installed_version, best_candidate.version)
        else:
            logger.debug('Existing installed version (%s) is most up-to-date and satisfies requirement', installed_version)
        return None
    if _should_install_candidate(best_candidate):
        logger.debug('Using version %s (newest of versions: %s)', best_candidate.version, _format_versions(best_candidate_result.iter_applicable()))
        return best_candidate
    logger.debug('Installed version (%s) is most up-to-date (past versions: %s)', installed_version, _format_versions(best_candidate_result.iter_applicable()))
    raise BestVersionAlreadyInstalled