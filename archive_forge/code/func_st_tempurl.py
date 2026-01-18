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
def st_tempurl(parser, args, thread_manager, return_parser=False):
    parser.add_argument('--absolute', action='store_true', dest='absolute_expiry', default=False, help='If present, and time argument is an integer, time argument will be interpreted as a Unix timestamp representing when the temporary URL should expire, rather than an offset from the current time.')
    parser.add_argument('--prefix-based', action='store_true', default=False, help='If present, a prefix-based temporary URL will be generated.')
    parser.add_argument('--iso8601', action='store_true', default=False, help='If present, the temporary URL will contain an ISO 8601 UTC timestamp instead of a Unix timestamp.')
    parser.add_argument('--ip-range', action='store', default=None, help='If present, the temporary URL will be restricted to the given ip or ip range.')
    parser.add_argument('--digest', choices=('sha1', 'sha256', 'sha512'), default='sha256', help='The digest algorithm to use. Defaults to sha256, but older clusters may only support sha1.')
    if return_parser:
        return parser
    options, args = parse_args(parser, args)
    args = args[1:]
    if len(args) < 4:
        thread_manager.error('Usage: %s tempurl %s\n%s', BASENAME, st_tempurl_options, st_tempurl_help)
        return
    method, timestamp, path, key = args[:4]
    parsed = urlparse(path)
    if method.upper() not in ['GET', 'PUT', 'HEAD', 'POST', 'DELETE']:
        thread_manager.print_msg('WARNING: Non default HTTP method %s for tempurl specified, possibly an error' % method.upper())
    try:
        path = generate_temp_url(parsed.path, timestamp, key, method, absolute=options['absolute_expiry'], iso8601=options['iso8601'], prefix=options['prefix_based'], ip_range=options['ip_range'], digest=options['digest'])
    except ValueError as err:
        thread_manager.error(err)
        return
    if parsed.scheme and parsed.netloc:
        url = '%s://%s%s' % (parsed.scheme, parsed.netloc, path)
    else:
        url = path
    thread_manager.print_msg(url)