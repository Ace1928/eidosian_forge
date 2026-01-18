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
def st_copy(parser, args, output_manager, return_parser=False):
    parser.add_argument('-d', '--destination', help='The container and name of the destination object')
    parser.add_argument('-M', '--fresh-metadata', action='store_true', help='Copy the object without any existing metadata', default=False)
    parser.add_argument('-m', '--meta', action='append', dest='meta', default=[], help='Sets a meta data item. This option may be repeated. Example: -m Color:Blue -m Size:Large')
    parser.add_argument('-H', '--header', action='append', dest='header', default=[], help='Adds a customized request header. This option may be repeated. Example: -H "content-type:text/plain" -H "Content-Length: 4000"')
    if return_parser:
        return parser
    options, args = parse_args(parser, args)
    args = args[1:]
    with SwiftService(options=options) as swift:
        try:
            if len(args) >= 2:
                container = args[0]
                if '/' in container:
                    output_manager.error("WARNING: / in container name; you might have meant '%s' instead of '%s'." % (args[0].replace('/', ' ', 1), args[0]))
                    return
                objects = [arg for arg in args[1:]]
                for r in swift.copy(container=container, objects=objects, options=options):
                    if r['success']:
                        if options['verbose']:
                            if r['action'] == 'copy_object':
                                output_manager.print_msg('%s/%s copied to %s' % (r['container'], r['object'], r['destination'] or '<self>'))
                            if r['action'] == 'create_container':
                                output_manager.print_msg('created container %s' % r['container'])
                    else:
                        error = r['error']
                        if 'action' in r and r['action'] == 'create_container':
                            output_manager.warning("Warning: failed to create container '%s': %s", container, error)
                        else:
                            output_manager.error('%s' % error)
            else:
                output_manager.error('Usage: %s copy %s\n%s', BASENAME, st_copy_options, st_copy_help)
                return
        except SwiftError as e:
            output_manager.error(e.value)