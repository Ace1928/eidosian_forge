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
@staticmethod
def determine_resolver_variant(options: Values) -> str:
    """Determines which resolver should be used, based on the given options."""
    if 'legacy-resolver' in options.deprecated_features_enabled:
        return 'legacy'
    return 'resolvelib'