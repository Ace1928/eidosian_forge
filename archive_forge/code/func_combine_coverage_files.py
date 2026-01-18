from __future__ import annotations
import collections.abc as c
import os
import json
import typing as t
from ...target import (
from ...io import (
from ...util import (
from ...util_common import (
from ...executor import (
from ...data import (
from ...host_configs import (
from ...provisioning import (
from . import (
def combine_coverage_files(args: CoverageCombineConfig, host_state: HostState) -> list[str]:
    """Combine coverage and return a list of the resulting files."""
    if args.delegate:
        if isinstance(args.controller, (DockerConfig, RemoteConfig)):
            paths = get_all_coverage_files()
            exported_paths = [path for path in paths if os.path.basename(path).split('=')[-1].split('.')[:2] == ['coverage', 'combined']]
            if not exported_paths:
                raise ExportedCoverageDataNotFound()
            pairs = [(path, os.path.relpath(path, data_context().content.root)) for path in exported_paths]

            def coverage_callback(payload_config: PayloadConfig) -> None:
                """Add the coverage files to the payload file list."""
                display.info('Including %d exported coverage file(s) in payload.' % len(pairs), verbosity=1)
                files = payload_config.files
                files.extend(pairs)
            data_context().register_payload_callback(coverage_callback)
        raise Delegate(host_state=host_state)
    paths = _command_coverage_combine_powershell(args) + _command_coverage_combine_python(args, host_state)
    for path in paths:
        display.info('Generated combined output: %s' % path, verbosity=1)
    return paths