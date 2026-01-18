import importlib.util
import logging
import os
import textwrap
from functools import partial
from optparse import SUPPRESS_HELP, Option, OptionGroup, OptionParser, Values
from textwrap import dedent
from typing import Any, Callable, Dict, Optional, Tuple
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.cli.parser import ConfigOptionParser
from pip._internal.exceptions import CommandError
from pip._internal.locations import USER_CACHE_DIR, get_src_prefix
from pip._internal.models.format_control import FormatControl
from pip._internal.models.index import PyPI
from pip._internal.models.target_python import TargetPython
from pip._internal.utils.hashes import STRONG_HASHES
from pip._internal.utils.misc import strtobool
def check_dist_restriction(options: Values, check_target: bool=False) -> None:
    """Function for determining if custom platform options are allowed.

    :param options: The OptionParser options.
    :param check_target: Whether or not to check if --target is being used.
    """
    dist_restriction_set = any([options.python_version, options.platforms, options.abis, options.implementation])
    binary_only = FormatControl(set(), {':all:'})
    sdist_dependencies_allowed = options.format_control != binary_only and (not options.ignore_dependencies)
    if dist_restriction_set and sdist_dependencies_allowed:
        raise CommandError('When restricting platform and interpreter constraints using --python-version, --platform, --abi, or --implementation, either --no-deps must be set, or --only-binary=:all: must be set and --no-binary must not be set (or must be set to :none:).')
    if check_target:
        if not options.dry_run and dist_restriction_set and (not options.target_dir):
            raise CommandError("Can not use any platform or abi specific options unless installing via '--target' or using '--dry-run'")