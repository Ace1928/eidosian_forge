import argparse
import sys
from IPython.core.magic_arguments import (argument, argument_group, kwds,
@magic_arguments()
@argument('-f', '--foo', help='an argument')
def magic_foo1(self, args):
    """ A docstring.
    """
    return parse_argstring(magic_foo1, args)