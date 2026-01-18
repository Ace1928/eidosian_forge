from __future__ import annotations
import argparse
import json
import logging
import time
from typing import Sequence
import flake8
from flake8 import checker
from flake8 import defaults
from flake8 import exceptions
from flake8 import style_guide
from flake8.formatting.base import BaseFormatter
from flake8.main import debug
from flake8.options.parse_args import parse_args
from flake8.plugins import finder
from flake8.plugins import reporter
def report_benchmarks(self) -> None:
    """Aggregate, calculate, and report benchmarks for this run."""
    assert self.options is not None
    if not self.options.benchmark:
        return
    assert self.file_checker_manager is not None
    assert self.end_time is not None
    time_elapsed = self.end_time - self.start_time
    statistics = [('seconds elapsed', time_elapsed)]
    add_statistic = statistics.append
    for statistic in defaults.STATISTIC_NAMES + ('files',):
        value = self.file_checker_manager.statistics[statistic]
        total_description = f'total {statistic} processed'
        add_statistic((total_description, value))
        per_second_description = f'{statistic} processed per second'
        add_statistic((per_second_description, int(value / time_elapsed)))
    assert self.formatter is not None
    self.formatter.show_benchmarks(statistics)