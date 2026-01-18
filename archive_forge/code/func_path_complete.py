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
def path_complete(self, text: str, line: str, begidx: int, endidx: int, *, path_filter: Optional[Callable[[str], bool]]=None) -> List[str]:
    """Performs completion of local file system paths

        :param text: the string prefix we are attempting to match (all matches must begin with it)
        :param line: the current input line with leading whitespace removed
        :param begidx: the beginning index of the prefix text
        :param endidx: the ending index of the prefix text
        :param path_filter: optional filter function that determines if a path belongs in the results
                            this function takes a path as its argument and returns True if the path should
                            be kept in the results
        :return: a list of possible tab completions
        """

    def complete_users() -> List[str]:
        users = []
        if sys.platform.startswith('win'):
            expanded_path = os.path.expanduser(text)
            if os.path.isdir(expanded_path):
                user = text
                if add_trailing_sep_if_dir:
                    user += os.path.sep
                users.append(user)
        else:
            import pwd
            for cur_pw in pwd.getpwall():
                if os.path.isdir(cur_pw.pw_dir):
                    cur_user = '~' + cur_pw.pw_name
                    if cur_user.startswith(text):
                        if add_trailing_sep_if_dir:
                            cur_user += os.path.sep
                        users.append(cur_user)
        if users:
            self.allow_appended_space = False
            self.allow_closing_quote = False
        return users
    add_trailing_sep_if_dir = False
    if endidx == len(line) or (endidx < len(line) and line[endidx] != os.path.sep):
        add_trailing_sep_if_dir = True
    cwd = os.getcwd()
    cwd_added = False
    orig_tilde_path = ''
    expanded_tilde_path = ''
    if not text:
        search_str = os.path.join(os.getcwd(), '*')
        cwd_added = True
    else:
        wildcards = ['*', '?']
        for wildcard in wildcards:
            if wildcard in text:
                return []
        search_str = text + '*'
        if text.startswith('~'):
            sep_index = text.find(os.path.sep, 1)
            if sep_index == -1:
                return complete_users()
            else:
                search_str = os.path.expanduser(search_str)
                orig_tilde_path = text[:sep_index]
                expanded_tilde_path = os.path.expanduser(orig_tilde_path)
        elif not os.path.dirname(text):
            search_str = os.path.join(os.getcwd(), search_str)
            cwd_added = True
    matches = glob.glob(search_str)
    if path_filter is not None:
        matches = [c for c in matches if path_filter(c)]
    if matches:
        self.matches_delimited = True
        if len(matches) == 1 and os.path.isdir(matches[0]):
            self.allow_appended_space = False
            self.allow_closing_quote = False
        matches.sort(key=self.default_sort_key)
        self.matches_sorted = True
        for index, cur_match in enumerate(matches):
            self.display_matches.append(os.path.basename(cur_match))
            if os.path.isdir(cur_match) and add_trailing_sep_if_dir:
                matches[index] += os.path.sep
                self.display_matches[index] += os.path.sep
        if cwd_added:
            if cwd == os.path.sep:
                to_replace = cwd
            else:
                to_replace = cwd + os.path.sep
            matches = [cur_path.replace(to_replace, '', 1) for cur_path in matches]
        if expanded_tilde_path:
            matches = [cur_path.replace(expanded_tilde_path, orig_tilde_path, 1) for cur_path in matches]
    return matches