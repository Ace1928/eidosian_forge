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
def can_skip_file(self, data: CoverageData, fr: FileReporter, rootname: str) -> bool:
    """Can we skip reporting this file?

        `data` is a CoverageData object, `fr` is a `FileReporter`, and
        `rootname` is the name being used for the file.
        """
    m = Hasher()
    m.update(fr.source().encode('utf-8'))
    add_data_to_hash(data, fr.filename, m)
    this_hash = m.hexdigest()
    that_hash = self.file_hash(rootname)
    if this_hash == that_hash:
        return True
    else:
        self.set_file_hash(rootname, this_hash)
        return False