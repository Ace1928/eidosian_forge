from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
def split_word(word):
    point_in_word = len(word) + point - lexer.instream.tell()
    if isinstance(lexer.state, (str, bytes)) and lexer.state in lexer.whitespace:
        point_in_word += 1
    if point_in_word > len(word):
        debug('In trailing whitespace')
        words.append(word)
        word = ''
    prefix, suffix = (word[:point_in_word], word[point_in_word:])
    prequote = ''
    if lexer.state is not None and lexer.state in lexer.quotes:
        prequote = lexer.state
    return (prequote, prefix, suffix, words, lexer.last_wordbreak_pos)