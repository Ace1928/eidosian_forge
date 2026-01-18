import argparse
import getpass
import io
import json
import logging
import signal
import socket
import warnings
from os import environ, walk, _exit as os_exit
from os.path import isfile, isdir, join
from urllib.parse import unquote, urlparse
from sys import argv as sys_argv, exit, stderr, stdin
from time import gmtime, strftime
from swiftclient import RequestException
from swiftclient.utils import config_true_value, generate_temp_url, \
from swiftclient.multithreading import OutputManager
from swiftclient.exceptions import ClientException
from swiftclient import __version__ as client_version
from swiftclient.client import logger_settings as client_logger_settings, \
from swiftclient.service import SwiftService, SwiftError, \
from swiftclient.command_helpers import print_account_stats, \
def st_bash_completion(parser, args, thread_manager, return_parser=False):
    if return_parser:
        return parser
    global commands
    com = args[1] if len(args) > 1 else None
    if com:
        if com in commands:
            fn_commands = ['st_%s' % com]
        else:
            print('')
            return
    else:
        fn_commands = [fn for fn in globals().keys() if fn.startswith('st_') and (not fn.endswith('_options')) and (not fn.endswith('_help'))]
    subparsers = parser.add_subparsers()
    subcommands = {}
    if not com:
        subcommands['base'] = parser
    for command in fn_commands:
        cmd = command[3:]
        if com:
            subparser = subparsers.add_parser(cmd, help=globals()['%s_help' % command])
            add_default_args(subparser)
            subparser = globals()[command](subparser, args, thread_manager, True)
            subcommands[cmd] = subparser
        else:
            subcommands[cmd] = None
    cmds = set()
    opts = set()
    for sc_str, sc in list(subcommands.items()):
        cmds.add(sc_str)
        if sc:
            for option in sc._optionals._option_string_actions:
                opts.add(option)
    for cmd_to_remove in (com, 'bash_completion', 'base'):
        if cmd_to_remove in cmds:
            cmds.remove(cmd_to_remove)
    print(' '.join(cmds | opts))