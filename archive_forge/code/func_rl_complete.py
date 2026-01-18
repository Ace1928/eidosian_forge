from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
def rl_complete(self, text, state):
    """
        Alternate entry point for using the argcomplete completer in a readline-based REPL. See also
        `rlcompleter <https://docs.python.org/2/library/rlcompleter.html#completer-objects>`_.
        Usage:

        .. code-block:: python

            import argcomplete, argparse, readline
            parser = argparse.ArgumentParser()
            ...
            completer = argcomplete.CompletionFinder(parser)
            readline.set_completer_delims("")
            readline.set_completer(completer.rl_complete)
            readline.parse_and_bind("tab: complete")
            result = input("prompt> ")

        (Use ``raw_input`` instead of ``input`` on Python 2, or use `eight <https://github.com/kislyuk/eight>`_).
        """
    if state == 0:
        cword_prequote, cword_prefix, cword_suffix, comp_words, first_colon_pos = split_line(text)
        comp_words.insert(0, sys.argv[0])
        matches = self._get_completions(comp_words, cword_prefix, cword_prequote, first_colon_pos)
        self._rl_matches = [text + match[len(cword_prefix):] for match in matches]
    if state < len(self._rl_matches):
        return self._rl_matches[state]
    else:
        return None