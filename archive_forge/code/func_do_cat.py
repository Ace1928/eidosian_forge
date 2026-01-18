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
def do_cat(self, test_case, input, args):
    in_name, out_name, out_mode, args = _scan_redirection_options(args)
    if args and in_name is not None:
        raise SyntaxError('Specify a file OR use redirection')
    inputs = []
    if input:
        inputs.append(input)
    input_names = args
    if in_name:
        args.append(in_name)
    for in_name in input_names:
        try:
            inputs.append(self._read_input(None, in_name))
        except OSError as e:
            if e.errno in (errno.ENOENT, errno.EINVAL):
                return (1, None, '{}: No such file or directory\n'.format(in_name))
            raise
    output = ''.join(inputs)
    try:
        output = self._write_output(output, out_name, out_mode)
    except OSError as e:
        if e.errno in (errno.ENOENT, errno.EINVAL):
            return (1, None, '{}: No such file or directory\n'.format(out_name))
        raise
    return (0, output, None)