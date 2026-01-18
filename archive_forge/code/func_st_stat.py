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
def st_stat(parser, args, output_manager, return_parser=False):
    parser.add_argument('--lh', dest='human', action='store_true', default=False, help='Report sizes in human readable format similar to ls -lh.')
    parser.add_argument('--version-id', action='store', default=None, help='Report stat of a specific version of a versioned object')
    parser.add_argument('-H', '--header', action='append', dest='header', default=[], help='Adds a custom request header to use for stat.')
    if return_parser:
        return parser
    options, args = parse_args(parser, args)
    args = args[1:]
    if options['version_id'] and len(args) < 2:
        exit('--version-id option only allowed for object stats')
    with SwiftService(options=options) as swift:
        try:
            if not args:
                stat_result = swift.stat()
                if not stat_result['success']:
                    raise stat_result['error']
                items = stat_result['items']
                headers = stat_result['headers']
                print_account_stats(items, headers, output_manager)
            else:
                container = args[0]
                if '/' in container:
                    output_manager.error("WARNING: / in container name; you might have meant '%s' instead of '%s'." % (container.replace('/', ' ', 1), container))
                    return
                args = args[1:]
                if not args:
                    stat_result = swift.stat(container=container)
                    if not stat_result['success']:
                        raise stat_result['error']
                    items = stat_result['items']
                    headers = stat_result['headers']
                    print_container_stats(items, headers, output_manager)
                elif len(args) == 1:
                    objects = [args[0]]
                    stat_results = swift.stat(container=container, objects=objects)
                    for stat_result in stat_results:
                        if stat_result['success']:
                            items = stat_result['items']
                            headers = stat_result['headers']
                            print_object_stats(items, headers, output_manager)
                        else:
                            raise stat_result['error']
                else:
                    output_manager.error('Usage: %s stat %s\n%s', BASENAME, st_stat_options, st_stat_help)
        except SwiftError as e:
            output_manager.error(e.value)