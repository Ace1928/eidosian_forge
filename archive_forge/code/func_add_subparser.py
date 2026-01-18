import argparse
import sys
def add_subparser(name, **args):
    """
    Add a subparser to the 'pyomo' command.
    """
    if _pyomo_subparsers is None:
        get_parser()
    func = args.pop('func', None)
    parser = _pyomo_subparsers.add_parser(name, **args)
    subparsers.append(name)
    if func is not None:
        parser.set_defaults(func=func)
    return parser