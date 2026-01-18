import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def index_based_complete(self, text: str, line: str, begidx: int, endidx: int, index_dict: Mapping[int, Union[Iterable[str], CompleterFunc]], *, all_else: Optional[Union[Iterable[str], CompleterFunc]]=None) -> List[str]:
    """Tab completes based on a fixed position in the input string.

        :param text: the string prefix we are attempting to match (all matches must begin with it)
        :param line: the current input line with leading whitespace removed
        :param begidx: the beginning index of the prefix text
        :param endidx: the ending index of the prefix text
        :param index_dict: dictionary whose structure is the following:
                           `keys` - 0-based token indexes into command line that determine which tokens perform tab
                           completion
                           `values` - there are two types of values:
                           1. iterable list of strings to match against (dictionaries, lists, etc.)
                           2. function that performs tab completion (ex: path_complete)
        :param all_else: an optional parameter for tab completing any token that isn't at an index in index_dict
        :return: a list of possible tab completions
        """
    tokens, _ = self.tokens_for_completion(line, begidx, endidx)
    if not tokens:
        return []
    matches = []
    index = len(tokens) - 1
    match_against: Optional[Union[Iterable[str], CompleterFunc]]
    if index in index_dict:
        match_against = index_dict[index]
    else:
        match_against = all_else
    if isinstance(match_against, Iterable):
        matches = self.basic_complete(text, line, begidx, endidx, match_against)
    elif callable(match_against):
        matches = match_against(text, line, begidx, endidx)
    return matches