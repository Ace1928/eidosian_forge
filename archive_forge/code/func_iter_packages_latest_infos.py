import json
import logging
from optparse import Values
from typing import TYPE_CHECKING, Generator, List, Optional, Sequence, Tuple, cast
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.cli import cmdoptions
from pip._internal.cli.req_command import IndexGroupCommand
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.exceptions import CommandError
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution, get_environment
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.network.session import PipSession
from pip._internal.utils.compat import stdlib_pkgs
from pip._internal.utils.misc import tabulate, write_output
def iter_packages_latest_infos(self, packages: '_ProcessedDists', options: Values) -> Generator['_DistWithLatestInfo', None, None]:
    with self._build_session(options) as session:
        finder = self._build_package_finder(options, session)

        def latest_info(dist: '_DistWithLatestInfo') -> Optional['_DistWithLatestInfo']:
            all_candidates = finder.find_all_candidates(dist.canonical_name)
            if not options.pre:
                all_candidates = [candidate for candidate in all_candidates if not candidate.version.is_prerelease]
            evaluator = finder.make_candidate_evaluator(project_name=dist.canonical_name)
            best_candidate = evaluator.sort_best_candidate(all_candidates)
            if best_candidate is None:
                return None
            remote_version = best_candidate.version
            if best_candidate.link.is_wheel:
                typ = 'wheel'
            else:
                typ = 'sdist'
            dist.latest_version = remote_version
            dist.latest_filetype = typ
            return dist
        for dist in map(latest_info, packages):
            if dist is not None:
                yield dist