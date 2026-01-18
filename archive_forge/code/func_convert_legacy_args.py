from __future__ import annotations
import argparse
import collections.abc as c
import dataclasses
import enum
import os
import types
import typing as t
from ..constants import (
from ..util import (
from ..docker_util import (
from ..completion import (
from ..host_configs import (
from ..data import (
def convert_legacy_args(argv: list[str], args: t.Union[argparse.Namespace, types.SimpleNamespace], mode: TargetMode) -> HostSettings:
    """Convert pre-split host arguments in the given namespace to their split counterparts."""
    old_options = LegacyHostOptions.create(args)
    old_options.purge_namespace(args)
    new_options = ['--controller', '--target', '--target-python', '--target-posix', '--target-windows', '--target-network']
    used_old_options = old_options.get_options_used()
    used_new_options = [name for name in new_options if name in argv]
    if used_old_options:
        if used_new_options:
            raise OptionsConflictError(used_old_options, used_new_options)
        controller, targets, controller_fallback = get_legacy_host_config(mode, old_options)
        if controller_fallback:
            if mode.one_host:
                display.info(controller_fallback.message, verbosity=1)
            else:
                display.warning(controller_fallback.message)
        used_default_pythons = mode in (TargetMode.SANITY, TargetMode.UNITS) and (not native_python(old_options))
    else:
        controller = args.controller or OriginConfig()
        controller_fallback = None
        if mode == TargetMode.NO_TARGETS:
            targets = []
            used_default_pythons = False
        elif args.targets:
            targets = args.targets
            used_default_pythons = False
        else:
            targets = default_targets(mode, controller)
            used_default_pythons = mode in (TargetMode.SANITY, TargetMode.UNITS)
    args.controller = controller
    args.targets = targets
    if used_default_pythons:
        control_targets = t.cast(list[ControllerConfig], targets)
        skipped_python_versions = sorted_versions(list(set(SUPPORTED_PYTHON_VERSIONS) - {target.python.version for target in control_targets}))
    else:
        skipped_python_versions = []
    filtered_args = old_options.purge_args(argv)
    filtered_args = filter_args(filtered_args, {name: 1 for name in new_options})
    host_settings = HostSettings(controller=controller, targets=targets, skipped_python_versions=skipped_python_versions, filtered_args=filtered_args, controller_fallback=controller_fallback)
    return host_settings