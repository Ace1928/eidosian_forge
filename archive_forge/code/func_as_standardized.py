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
def as_standardized(self) -> 'CallgrindStats':
    """Strip library names and some prefixes from function strings.

        When comparing two different sets of instruction counts, on stumbling
        block can be path prefixes. Callgrind includes the full filepath
        when reporting a function (as it should). However, this can cause
        issues when diffing profiles. If a key component such as Python
        or PyTorch was built in separate locations in the two profiles, which
        can result in something resembling::

            23234231 /tmp/first_build_dir/thing.c:foo(...)
             9823794 /tmp/first_build_dir/thing.c:bar(...)
              ...
               53453 .../aten/src/Aten/...:function_that_actually_changed(...)
              ...
             -9823794 /tmp/second_build_dir/thing.c:bar(...)
            -23234231 /tmp/second_build_dir/thing.c:foo(...)

        Stripping prefixes can ameliorate this issue by regularizing the
        strings and causing better cancellation of equivalent call sites
        when diffing.
        """

    def strip(stats: FunctionCounts) -> FunctionCounts:
        transforms = (('^.+build/\\.\\./', 'build/../'), ('^.+/' + re.escape('build/aten/'), 'build/aten/'), ('^.+/' + re.escape('Python/'), 'Python/'), ('^.+/' + re.escape('Objects/'), 'Objects/'), ('\\s\\[.+\\]$', ''))
        for before, after in transforms:
            stats = stats.transform(lambda fn: re.sub(before, after, fn))
        return stats
    return CallgrindStats(task_spec=self.task_spec, number_per_run=self.number_per_run, built_with_debug_symbols=self.built_with_debug_symbols, baseline_inclusive_stats=strip(self.baseline_inclusive_stats), baseline_exclusive_stats=strip(self.baseline_exclusive_stats), stmt_inclusive_stats=strip(self.stmt_inclusive_stats), stmt_exclusive_stats=strip(self.stmt_exclusive_stats), stmt_callgrind_out=None)