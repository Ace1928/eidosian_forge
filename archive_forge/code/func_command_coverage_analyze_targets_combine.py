from __future__ import annotations
import typing as t
from .....executor import (
from .....provisioning import (
from . import (
from . import (
def command_coverage_analyze_targets_combine(args: CoverageAnalyzeTargetsCombineConfig) -> None:
    """Combine integration test target code coverage reports."""
    host_state = prepare_profiles(args)
    if args.delegate:
        raise Delegate(host_state=host_state)
    combined_target_indexes: TargetIndexes = {}
    combined_path_arcs: Arcs = {}
    combined_path_lines: Lines = {}
    for report_path in args.input_files:
        covered_targets, covered_path_arcs, covered_path_lines = read_report(report_path)
        merge_indexes(covered_path_arcs, covered_targets, combined_path_arcs, combined_target_indexes)
        merge_indexes(covered_path_lines, covered_targets, combined_path_lines, combined_target_indexes)
    report = make_report(combined_target_indexes, combined_path_arcs, combined_path_lines)
    write_report(args, report, args.output_file)