from __future__ import annotations
import abc
import argparse
import os
import re
import typing as t
from .argcompletion import (
from .parsers import (
def detect_false_file_completion(value: str, mode: ParserMode) -> bool:
    """
    Return True if Bash will provide an incorrect file completion, otherwise return False.

    If there are no completion results, a filename will be automatically completed if the value after the last `=` or `:` character:

        - matches exactly one partial path

    Otherwise Bash will play the bell sound and display nothing.

    see: https://github.com/kislyuk/argcomplete/issues/328
    see: https://github.com/kislyuk/argcomplete/pull/284
    """
    completion = False
    if mode == ParserMode.COMPLETE:
        completion = True
        right = re.split('[=:]', value)[-1]
        directory, prefix = os.path.split(right)
        try:
            filenames = os.listdir(directory or '.')
        except Exception:
            pass
        else:
            matches = [filename for filename in filenames if filename.startswith(prefix)]
            completion = len(matches) == 1
    return completion