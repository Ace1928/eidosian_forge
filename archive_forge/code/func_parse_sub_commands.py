import pkg_resources
import argparse
import logging
import sys
from warnings import warn
def parse_sub_commands(self):
    subparsers = self.parser.add_subparsers(dest='command_name', metavar='command')
    for name, cmd in self.commands.items():
        sub = subparsers.add_parser(name, help=cmd.summary)
        for arg in getattr(cmd, 'arguments', tuple()):
            arg = arg.copy()
            if isinstance(arg.get('name'), str):
                sub.add_argument(arg.pop('name'), **arg)
            elif isinstance(arg.get('name'), list):
                sub.add_argument(*arg.pop('name'), **arg)