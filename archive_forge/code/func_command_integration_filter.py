from __future__ import annotations
import collections.abc as c
import contextlib
import datetime
import json
import os
import re
import shutil
import tempfile
import time
import typing as t
from ...encoding import (
from ...ansible_util import (
from ...executor import (
from ...python_requirements import (
from ...ci import (
from ...target import (
from ...config import (
from ...io import (
from ...util import (
from ...util_common import (
from ...coverage_util import (
from ...cache import (
from .cloud import (
from ...data import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...inventory import (
from .filters import (
from .coverage import (
def command_integration_filter(args: TIntegrationConfig, targets: c.Iterable[TIntegrationTarget]) -> tuple[HostState, tuple[TIntegrationTarget, ...]]:
    """Filter the given integration test targets."""
    targets = tuple((target for target in targets if 'hidden/' not in target.aliases))
    changes = get_changes_filter(args)
    if args.changed_all_target in changes:
        if args.changed_all_mode == 'include' and args.changed_all_target not in args.include:
            args.include.append(args.changed_all_target)
            args.delegate_args += ['--include', args.changed_all_target]
        elif args.changed_all_mode == 'exclude' and args.changed_all_target not in args.exclude:
            args.exclude.append(args.changed_all_target)
    require = args.require + changes
    exclude = args.exclude
    internal_targets = walk_internal_targets(targets, args.include, exclude, require)
    environment_exclude = get_integration_filter(args, list(internal_targets))
    environment_exclude |= set(cloud_filter(args, internal_targets))
    if environment_exclude:
        exclude = sorted(set(exclude) | environment_exclude)
        internal_targets = walk_internal_targets(targets, args.include, exclude, require)
    if not internal_targets:
        raise AllTargetsSkipped()
    if args.start_at and (not any((target.name == args.start_at for target in internal_targets))):
        raise ApplicationError('Start at target matches nothing: %s' % args.start_at)
    cloud_init(args, internal_targets)
    vars_file_src = os.path.join(data_context().content.root, data_context().content.integration_vars_path)
    if os.path.exists(vars_file_src):

        def integration_config_callback(payload_config: PayloadConfig) -> None:
            """
            Add the integration config vars file to the payload file list.
            This will preserve the file during delegation even if the file is ignored by source control.
            """
            files = payload_config.files
            files.append((vars_file_src, data_context().content.integration_vars_path))
        data_context().register_payload_callback(integration_config_callback)
    if args.list_targets:
        raise ListTargets([target.name for target in internal_targets])
    host_state = prepare_profiles(args, targets_use_pypi=True, requirements=requirements)
    if args.delegate:
        raise Delegate(host_state=host_state, require=require, exclude=exclude)
    return (host_state, internal_targets)