from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def run_completer(self, parser, completer, command, point=None, **kwargs):
    cword_prequote, cword_prefix, cword_suffix, comp_words, first_colon_pos = split_line(command)
    completions = completer._get_completions(comp_words, cword_prefix, cword_prequote, first_colon_pos)
    return completions