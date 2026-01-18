from __future__ import annotations
import os
import typing as t
from .....encoding import (
from .....data import (
from .....util_common import (
from .....executor import (
from .....provisioning import (
from ... import (
from . import (
from . import (
def command_coverage_analyze_targets_generate(args: CoverageAnalyzeTargetsGenerateConfig) -> None:
    """Analyze code coverage data to determine which integration test targets provide coverage for each arc or line."""
    host_state = prepare_profiles(args)
    if args.delegate:
        raise Delegate(host_state)
    root = data_context().content.root
    target_indexes: TargetIndexes = {}
    arcs = dict(((os.path.relpath(path, root), data) for path, data in analyze_python_coverage(args, host_state, args.input_dir, target_indexes).items()))
    lines = dict(((os.path.relpath(path, root), data) for path, data in analyze_powershell_coverage(args, args.input_dir, target_indexes).items()))
    report = make_report(target_indexes, arcs, lines)
    write_report(args, report, args.output_file)