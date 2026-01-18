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
def st_capabilities(parser, args, output_manager, return_parser=False):

    def _print_compo_cap(name, capabilities):
        for feature, options in sorted(capabilities.items(), key=lambda x: x[0]):
            output_manager.print_msg('%s: %s' % (name, feature))
            if options:
                output_manager.print_msg(' Options:')
                for key, value in sorted(options.items(), key=lambda x: x[0]):
                    output_manager.print_msg('  %s: %s' % (key, value))
    parser.add_argument('--json', action='store_true', help='print capability information in json')
    if return_parser:
        return parser
    options, args = parse_args(parser, args)
    if args and len(args) > 2:
        output_manager.error('Usage: %s capabilities %s\n%s', BASENAME, st_capabilities_options, st_capabilities_help)
        return
    with SwiftService(options=options) as swift:
        try:
            if len(args) == 2:
                url = args[1]
                capabilities_result = swift.capabilities(url)
                capabilities = capabilities_result['capabilities']
            else:
                capabilities_result = swift.capabilities()
                capabilities = capabilities_result['capabilities']
            if options['json']:
                output_manager.print_msg(json.dumps(capabilities, sort_keys=True, indent=2))
            else:
                capabilities = dict(capabilities)
                _print_compo_cap('Core', {'swift': capabilities['swift']})
                del capabilities['swift']
                _print_compo_cap('Additional middleware', capabilities)
        except SwiftError as e:
            output_manager.error(e.value)