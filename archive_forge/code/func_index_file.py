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
def index_file(self, first_html: str, final_html: str) -> None:
    """Write the index.html file for this report."""
    self.make_directory()
    index_tmpl = Templite(read_data('index.html'), self.template_globals)
    skipped_covered_msg = skipped_empty_msg = ''
    if self.skipped_covered_count:
        n = self.skipped_covered_count
        skipped_covered_msg = f'{n} file{plural(n)} skipped due to complete coverage.'
    if self.skipped_empty_count:
        n = self.skipped_empty_count
        skipped_empty_msg = f'{n} empty file{plural(n)} skipped.'
    html = index_tmpl.render({'files': self.file_summaries, 'totals': self.totals, 'skipped_covered_msg': skipped_covered_msg, 'skipped_empty_msg': skipped_empty_msg, 'first_html': first_html, 'final_html': final_html})
    index_file = os.path.join(self.directory, 'index.html')
    write_html(index_file, html)
    print_href = stdout_link(index_file, f'file://{os.path.abspath(index_file)}')
    self.coverage._message(f'Wrote HTML report to {print_href}')
    self.incr.write()