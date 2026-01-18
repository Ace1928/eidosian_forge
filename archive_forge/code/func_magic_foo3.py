import argparse
import sys
from IPython.core.magic_arguments import (argument, argument_group, kwds,
@magic_arguments()
@argument('-f', '--foo', help='an argument')
@argument_group('Group')
@argument('-b', '--bar', help='a grouped argument')
@argument_group('Second Group')
@argument('-z', '--baz', help='another grouped argument')
def magic_foo3(self, args):
    """ A docstring.
    """
    return parse_argstring(magic_foo3, args)