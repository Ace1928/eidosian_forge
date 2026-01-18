from __future__ import annotations
import sys
from typing import Any, IO, Iterable, TYPE_CHECKING
from coverage.exceptions import ConfigError, NoDataError
from coverage.misc import human_sorted_items
from coverage.plugin import FileReporter
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.types import TMorf
def report_one_file(self, fr: FileReporter, analysis: Analysis) -> None:
    """Report on just one file, the callback from report()."""
    nums = analysis.numbers
    self.total += nums
    no_missing_lines = nums.n_missing == 0
    no_missing_branches = nums.n_partial_branches == 0
    if self.config.skip_covered and no_missing_lines and no_missing_branches:
        self.skipped_count += 1
    elif self.config.skip_empty and nums.n_statements == 0:
        self.empty_count += 1
    else:
        self.fr_analysis.append((fr, analysis))