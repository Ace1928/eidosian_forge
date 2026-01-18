from __future__ import annotations
import argparse
from ..target import (
from .argparsing.argcompletion import (
def complete_choices(choices: list[str], prefix: str, **_) -> list[str]:
    """Perform completion using the provided choices."""
    matches = [choice for choice in choices if choice.startswith(prefix)]
    return matches