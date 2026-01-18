from __future__ import annotations
import argparse
import logging
import os.path
from typing import Any
from flake8.discover_files import expand_paths
from flake8.formatting import base as formatter
from flake8.main import application as app
from flake8.options.parse_args import parse_args
def excluded(path: str) -> bool:
    paths = tuple(expand_paths(paths=[path], stdin_display_name=self.options.stdin_display_name, filename_patterns=self.options.filename, exclude=self.options.exclude))
    return not paths