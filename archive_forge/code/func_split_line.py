from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, contextlib
from . import completers, my_shlex as shlex
from .compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
from .completers import FilesCompleter, SuppressCompleter
from .my_argparse import IntrospectiveArgumentParser, action_is_satisfied, action_is_open, action_is_greedy
from .shellintegration import shellcode # noqa
def split_line(line, point=None):
    if point is None:
        point = len(line)
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.wordbreaks = os.environ.get('_ARGCOMPLETE_COMP_WORDBREAKS', '')
    words = []

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
    while True:
        try:
            word = lexer.get_token()
            if word == lexer.eof:
                return ('', '', '', words, None)
            if lexer.instream.tell() >= point:
                debug('word', word, "split, lexer state: '{s}'".format(s=lexer.state))
                return split_word(word)
            words.append(word)
        except ValueError:
            debug('word', lexer.token, "split (lexer stopped, state: '{s}')".format(s=lexer.state))
            if lexer.instream.tell() >= point:
                return split_word(lexer.token)
            else:
                raise ArgcompleteException('Unexpected internal state. Please report this bug at https://github.com/kislyuk/argcomplete/issues.')