from __future__ import annotations
import logging # isort:skip
import argparse
import sys
from abc import abstractmethod
from os.path import splitext
from ...document import Document
from ..subcommand import (
from ..util import build_single_handler_applications, die
def after_write_file(self, args: argparse.Namespace, filename: str, doc: Document) -> None:
    """

        """
    pass