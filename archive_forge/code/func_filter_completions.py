from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
def filter_completions(self, completions):
    """
        Ensures collected completions are Unicode text, de-duplicates them, and excludes those specified by ``exclude``.
        Returns the filtered completions as an iterable.

        This method is exposed for overriding in subclasses; there is no need to use it directly.
        """
    completions = [ensure_str(c) for c in completions]
    if self.exclude is None:
        self.exclude = set()
    seen = set(self.exclude)
    return [c for c in completions if c not in seen and (not seen.add(c))]