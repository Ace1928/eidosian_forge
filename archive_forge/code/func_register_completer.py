from __future__ import annotations
import argparse
from ..target import (
from .argparsing.argcompletion import (
def register_completer(action: argparse.Action, completer) -> None:
    """Register the given completer with the specified action."""
    action.completer = completer