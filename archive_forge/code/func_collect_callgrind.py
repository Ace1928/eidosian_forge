import collections
import enum
import dataclasses
import itertools as it
import os
import pickle
import re
import shutil
import subprocess
import sys
import textwrap
from typing import (
import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import CallgrindModuleType
def collect_callgrind(self, task_spec: common.TaskSpec, globals: Dict[str, Any], *, number: int, repeats: int, collect_baseline: bool, is_python: bool, retain_out_file: bool) -> Tuple[CallgrindStats, ...]:
    """Collect stats, and attach a reference run which can be used to filter interpreter overhead."""
    self._validate()
    assert is_python or not collect_baseline
    *task_stats, baseline_stats = self._invoke(task_spec=task_spec, globals=globals, number=number, repeats=repeats, collect_baseline=collect_baseline, is_python=is_python, retain_out_file=retain_out_file)
    assert len(task_stats) == repeats
    return tuple((CallgrindStats(task_spec=task_spec, number_per_run=number, built_with_debug_symbols=self._build_type == 'RelWithDebInfo', baseline_inclusive_stats=baseline_stats[0], baseline_exclusive_stats=baseline_stats[1], stmt_inclusive_stats=stmt_inclusive_stats, stmt_exclusive_stats=stmt_exclusive_stats, stmt_callgrind_out=out_contents) for stmt_inclusive_stats, stmt_exclusive_stats, out_contents in task_stats))