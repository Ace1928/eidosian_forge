import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
def do_echo(self, test_case, input, args):
    in_name, out_name, out_mode, args = _scan_redirection_options(args)
    if input or in_name:
        raise SyntaxError("echo doesn't read from stdin")
    if args:
        input = ' '.join(args)
    input += '\n'
    output = input
    try:
        output = self._write_output(output, out_name, out_mode)
    except OSError as e:
        if e.errno in (errno.ENOENT, errno.EINVAL):
            return (1, None, '{}: No such file or directory\n'.format(out_name))
        raise
    return (0, output, None)