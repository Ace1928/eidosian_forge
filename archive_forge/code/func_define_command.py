import argparse
import base64
import contextlib
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from oslo_utils import strutils
import yaml
from ironicclient.common.i18n import _
from ironicclient import exc
def define_command(subparsers, command, callback, cmd_mapper):
    """Define a command in the subparsers collection.

    :param subparsers: subparsers collection where the command will go
    :param command: command name
    :param callback: function that will be used to process the command
    """
    desc = callback.__doc__ or ''
    help = desc.strip().split('\n')[0]
    arguments = getattr(callback, 'arguments', [])
    subparser = subparsers.add_parser(command, help=help, description=desc, add_help=False, formatter_class=HelpFormatter)
    subparser.add_argument('-h', '--help', action='help', help=argparse.SUPPRESS)
    cmd_mapper[command] = subparser
    required_args = subparser.add_argument_group(_('Required arguments'))
    for args, kwargs in arguments:
        if kwargs.get('required'):
            required_args.add_argument(*args, **kwargs)
        else:
            subparser.add_argument(*args, **kwargs)
    subparser.set_defaults(func=callback)