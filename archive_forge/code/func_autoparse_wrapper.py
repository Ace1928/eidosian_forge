import sys
from re import compile as compile_regex
from inspect import signature, getdoc, Parameter
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import wraps
from io import IOBase
from autocommand.errors import AutocommandError
@wraps(func)
def autoparse_wrapper(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parsed_args = func_sig.bind_partial()
    parsed_args.arguments.update(vars(parser.parse_args(argv)))
    return func(*parsed_args.args, **parsed_args.kwargs)