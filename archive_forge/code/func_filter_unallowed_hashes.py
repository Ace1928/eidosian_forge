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
def filter_unallowed_hashes(candidates: List[InstallationCandidate], hashes: Optional[Hashes], project_name: str) -> List[InstallationCandidate]:
    """
    Filter out candidates whose hashes aren't allowed, and return a new
    list of candidates.

    If at least one candidate has an allowed hash, then all candidates with
    either an allowed hash or no hash specified are returned.  Otherwise,
    the given candidates are returned.

    Including the candidates with no hash specified when there is a match
    allows a warning to be logged if there is a more preferred candidate
    with no hash specified.  Returning all candidates in the case of no
    matches lets pip report the hash of the candidate that would otherwise
    have been installed (e.g. permitting the user to more easily update
    their requirements file with the desired hash).
    """
    if not hashes:
        logger.debug('Given no hashes to check %s links for project %r: discarding no candidates', len(candidates), project_name)
        return list(candidates)
    matches_or_no_digest = []
    non_matches = []
    match_count = 0
    for candidate in candidates:
        link = candidate.link
        if not link.has_hash:
            pass
        elif link.is_hash_allowed(hashes=hashes):
            match_count += 1
        else:
            non_matches.append(candidate)
            continue
        matches_or_no_digest.append(candidate)
    if match_count:
        filtered = matches_or_no_digest
    else:
        filtered = list(candidates)
    if len(filtered) == len(candidates):
        discard_message = 'discarding no candidates'
    else:
        discard_message = 'discarding {} non-matches:\n  {}'.format(len(non_matches), '\n  '.join((str(candidate.link) for candidate in non_matches)))
    logger.debug('Checked %s links for project %r against %s hashes (%s matches, %s no digest): %s', len(candidates), project_name, hashes.digest_count, match_count, len(matches_or_no_digest) - match_count, discard_message)
    return filtered