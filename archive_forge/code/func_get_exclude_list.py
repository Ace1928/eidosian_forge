from __future__ import annotations
import atexit
import collections
import contextlib
import os
import os.path
import platform
import signal
import sys
import threading
import time
import warnings
from types import FrameType
from typing import (
from coverage import env
from coverage.annotate import AnnotateReporter
from coverage.collector import Collector, HAS_CTRACER
from coverage.config import CoverageConfig, read_coverage_config
from coverage.context import should_start_context_test_function, combine_context_switchers
from coverage.data import CoverageData, combine_parallel_data
from coverage.debug import (
from coverage.disposition import disposition_debug_msg
from coverage.exceptions import ConfigError, CoverageException, CoverageWarning, PluginError
from coverage.files import PathAliases, abs_file, relative_filename, set_relative_directory
from coverage.html import HtmlReporter
from coverage.inorout import InOrOut
from coverage.jsonreport import JsonReporter
from coverage.lcovreport import LcovReporter
from coverage.misc import bool_or_none, join_regex
from coverage.misc import DefaultValue, ensure_dir_for_file, isolate_module
from coverage.multiproc import patch_multiprocessing
from coverage.plugin import FileReporter
from coverage.plugin_support import Plugins
from coverage.python import PythonFileReporter
from coverage.report import SummaryReporter
from coverage.report_core import render_report
from coverage.results import Analysis
from coverage.types import (
from coverage.xmlreport import XmlReporter
def get_exclude_list(self, which: str='exclude') -> list[str]:
    """Return a list of excluded regex strings.

        `which` indicates which list is desired.  See :meth:`exclude` for the
        lists that are available, and their meaning.

        """
    self._init()
    return cast(List[str], getattr(self.config, which + '_list'))