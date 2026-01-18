import argparse
import os
import sys
import urllib.parse  # noqa: WPS301
from importlib import import_module
from contextlib import suppress
from . import server
from . import wsgi
def server_args(self, parsed_args):
    """Return keyword args for Server class."""
    args = {arg: value for arg, value in vars(parsed_args).items() if not arg.startswith('_') and value is not None}
    args.update(vars(self))
    return args