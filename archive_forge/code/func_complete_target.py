from __future__ import annotations
import argparse
from ..target import (
from .argparsing.argcompletion import (
def complete_target(completer: OptionCompletionFinder, prefix: str, parsed_args: argparse.Namespace, **_) -> list[str]:
    """Perform completion for the targets configured for the command being parsed."""
    matches = find_target_completion(parsed_args.targets_func, prefix, completer.list_mode)
    completer.disable_completion_mangling = completer.list_mode and len(matches) > 1
    return matches