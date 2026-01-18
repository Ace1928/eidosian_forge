import argparse
import sys
from IPython.core.magic_arguments import (argument, argument_group, kwds,
@magic_arguments()
@kwds(argument_default=argparse.SUPPRESS)
@argument('-f', '--foo', help='an argument')
def magic_foo4(self, args):
    """ A docstring.
    """
    return parse_argstring(magic_foo4, args)