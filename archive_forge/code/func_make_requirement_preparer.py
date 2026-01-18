import logging
import os
import sys
from functools import partial
from optparse import Values
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from pip._internal.cache import WheelCache
from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.command_context import CommandContextMixIn
from pip._internal.exceptions import CommandError, PreviousBuildDirError
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.models.target_python import TargetPython
from pip._internal.network.session import PipSession
from pip._internal.operations.build.build_tracker import BuildTracker
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.constructors import (
from pip._internal.req.req_file import parse_requirements
from pip._internal.req.req_install import InstallRequirement
from pip._internal.resolution.base import BaseResolver
from pip._internal.self_outdated_check import pip_self_version_check
from pip._internal.utils.temp_dir import (
from pip._internal.utils.virtualenv import running_under_virtualenv
@classmethod
def make_requirement_preparer(cls, temp_build_dir: TempDirectory, options: Values, build_tracker: BuildTracker, session: PipSession, finder: PackageFinder, use_user_site: bool, download_dir: Optional[str]=None, verbosity: int=0) -> RequirementPreparer:
    """
        Create a RequirementPreparer instance for the given parameters.
        """
    temp_build_dir_path = temp_build_dir.path
    assert temp_build_dir_path is not None
    legacy_resolver = False
    resolver_variant = cls.determine_resolver_variant(options)
    if resolver_variant == 'resolvelib':
        lazy_wheel = 'fast-deps' in options.features_enabled
        if lazy_wheel:
            logger.warning('pip is using lazily downloaded wheels using HTTP range requests to obtain dependency information. This experimental feature is enabled through --use-feature=fast-deps and it is not ready for production.')
    else:
        legacy_resolver = True
        lazy_wheel = False
        if 'fast-deps' in options.features_enabled:
            logger.warning('fast-deps has no effect when used with the legacy resolver.')
    return RequirementPreparer(build_dir=temp_build_dir_path, src_dir=options.src_dir, download_dir=download_dir, build_isolation=options.build_isolation, check_build_deps=options.check_build_deps, build_tracker=build_tracker, session=session, progress_bar=options.progress_bar, finder=finder, require_hashes=options.require_hashes, use_user_site=use_user_site, lazy_wheel=lazy_wheel, verbosity=verbosity, legacy_resolver=legacy_resolver)