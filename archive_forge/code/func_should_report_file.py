from __future__ import annotations
import collections
import datetime
import functools
import json
import os
import re
import shutil
import string
from dataclasses import dataclass
from typing import Any, Iterable, TYPE_CHECKING, cast
import coverage
from coverage.data import CoverageData, add_data_to_hash
from coverage.exceptions import NoDataError
from coverage.files import flat_rootname
from coverage.misc import ensure_dir, file_be_gone, Hasher, isolate_module, format_local_datetime
from coverage.misc import human_sorted, plural, stdout_link
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.templite import Templite
from coverage.types import TLineNo, TMorf
from coverage.version import __url__
def should_report_file(self, ftr: FileToReport) -> bool:
    """Determine if we'll report this file."""
    nums = ftr.analysis.numbers
    self.all_files_nums.append(nums)
    if self.skip_covered:
        no_missing_lines = nums.n_missing == 0
        no_missing_branches = nums.n_partial_branches == 0
        if no_missing_lines and no_missing_branches:
            self.skipped_covered_count += 1
            return False
    if self.skip_empty:
        if nums.n_statements == 0:
            self.skipped_empty_count += 1
            return False
    return True