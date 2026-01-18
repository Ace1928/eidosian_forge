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
def analyze_python_coverage(args: CoverageAnalyzeTargetsGenerateConfig, host_state: HostState, path: str, target_indexes: TargetIndexes) -> Arcs:
    """Analyze Python code coverage."""
    results: Arcs = {}
    collection_search_re, collection_sub_re = get_collection_path_regexes()
    modules = get_python_modules()
    python_files = get_python_coverage_files(path)
    coverage = initialize_coverage(args, host_state)
    for python_file in python_files:
        if not is_integration_coverage_file(python_file):
            continue
        target_name = get_target_name(python_file)
        target_index = get_target_index(target_name, target_indexes)
        for filename, covered_arcs in enumerate_python_arcs(python_file, coverage, modules, collection_search_re, collection_sub_re):
            arcs = results.setdefault(filename, {})
            for covered_arc in covered_arcs:
                arc = arcs.setdefault(covered_arc, set())
                arc.add(target_index)
    prune_invalid_filenames(args, results, collection_search_re=collection_search_re)
    return results