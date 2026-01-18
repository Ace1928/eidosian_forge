import os
import os.path
import pathlib
import posixpath
import stat
import sys
import warnings
from collections.abc import (
from dataclasses import (
from os import (
from typing import (
from .pattern import (
def check_match_file(patterns: Iterable[Tuple[int, Pattern]], file: str) -> Tuple[Optional[bool], Optional[int]]:
    """
	Check the file against the patterns.

	*patterns* (:class:`~collections.abc.Iterable`) yields each indexed pattern
	(:class:`tuple`) which contains the pattern index (:class:`int`) and actual
	pattern (:class:`~pathspec.pattern.Pattern`).

	*file* (:class:`str`) is the normalized file path to be matched
	against *patterns*.

	Returns a :class:`tuple` containing whether to include *file* (:class:`bool`
	or :data:`None`), and the index of the last matched pattern (:class:`int` or
	:data:`None`).
	"""
    out_include: Optional[bool] = None
    out_index: Optional[int] = None
    for index, pattern in patterns:
        if pattern.include is not None and pattern.match_file(file) is not None:
            out_include = pattern.include
            out_index = index
    return (out_include, out_index)