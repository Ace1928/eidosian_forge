from __future__ import annotations
import argparse
import ast
import logging
from typing import Any
from typing import Generator
import pyflakes.checker
from flake8.options.manager import OptionManager
@classmethod
def parse_options(cls, options: argparse.Namespace) -> None:
    """Parse option values from Flake8's OptionManager."""
    if options.builtins:
        cls.builtIns = cls.builtIns.union(options.builtins)
    cls.with_doctest = options.doctests