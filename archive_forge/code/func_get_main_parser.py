import argparse
import logging
import os
import sys
from importlib import import_module
from . import SUPPORTED_SHELLS, __version__, add_argument_to, complete
def get_main_parser():
    parser = argparse.ArgumentParser(prog='shtab')
    parser.add_argument('parser', help='importable parser (or function returning parser)')
    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    parser.add_argument('-s', '--shell', default=SUPPORTED_SHELLS[0], choices=SUPPORTED_SHELLS)
    parser.add_argument('--prefix', help='prepended to generated functions to avoid clashes')
    parser.add_argument('--preamble', help='prepended to generated script')
    parser.add_argument('--prog', help='custom program name (overrides `parser.prog`)')
    parser.add_argument('-u', '--error-unimportable', default=False, action='store_true', help='raise errors if `parser` is not found in $PYTHONPATH')
    parser.add_argument('--verbose', dest='loglevel', action='store_const', default=logging.INFO, const=logging.DEBUG, help='Log debug information')
    add_argument_to(parser, '--print-own-completion', help="print shtab's own completion")
    return parser