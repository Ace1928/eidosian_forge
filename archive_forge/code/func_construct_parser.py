import argparse
import re
from IPython.core.error import UsageError
from IPython.utils.decorators import undoc
from IPython.utils.process import arg_split
from IPython.utils.text import dedent
def construct_parser(magic_func):
    """ Construct an argument parser using the function decorations.
    """
    kwds = getattr(magic_func, 'argcmd_kwds', {})
    if 'description' not in kwds:
        kwds['description'] = getattr(magic_func, '__doc__', None)
    arg_name = real_name(magic_func)
    parser = MagicArgumentParser(arg_name, **kwds)
    group = None
    for deco in magic_func.decorators[::-1]:
        result = deco.add_to_parser(parser, group)
        if result is not None:
            group = result
    magic_func.__doc__ = parser.format_help()
    return parser