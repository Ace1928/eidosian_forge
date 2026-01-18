import argparse
import sys
from IPython.core.magic_arguments import (argument, argument_group, kwds,
@magic_arguments()
@argument('-f', '--foo', help='an argument')
def magic_magic_foo(self, args):
    """ A docstring.
    """
    return parse_argstring(magic_magic_foo, args)