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
def st_post(parser, args, output_manager, return_parser=False):
    parser.add_argument('-r', '--read-acl', dest='read_acl', help='Read ACL for containers. Quick summary of ACL syntax: .r:*, .r:-.example.com, .r:www.example.com, account1, account2:user2')
    parser.add_argument('-w', '--write-acl', dest='write_acl', help='Write ACL for containers. Quick summary of ACL syntax: account1, account2:user2')
    parser.add_argument('-t', '--sync-to', dest='sync_to', help='Sets the Sync To for containers, for multi-cluster replication.')
    parser.add_argument('-k', '--sync-key', dest='sync_key', help='Sets the Sync Key for containers, for multi-cluster replication.')
    parser.add_argument('-m', '--meta', action='append', dest='meta', default=[], help='Sets a meta data item. This option may be repeated. Example: -m Color:Blue -m Size:Large')
    parser.add_argument('-H', '--header', action='append', dest='header', default=[], help='Adds a customized request header. This option may be repeated. Example: -H "content-type:text/plain" -H "Content-Length: 4000"')
    if return_parser:
        return parser
    options, args = parse_args(parser, args)
    args = args[1:]
    if (options['read_acl'] or options['write_acl'] or options['sync_to'] or options['sync_key']) and (not args):
        exit('-r, -w, -t, and -k options only allowed for containers')
    with SwiftService(options=options) as swift:
        try:
            if not args:
                result = swift.post()
            else:
                container = args[0]
                if '/' in container:
                    output_manager.error("WARNING: / in container name; you might have meant '%s' instead of '%s'." % (args[0].replace('/', ' ', 1), args[0]))
                    return
                args = args[1:]
                if args:
                    if len(args) == 1:
                        objects = [args[0]]
                        results_iterator = swift.post(container=container, objects=objects)
                        result = next(results_iterator)
                    else:
                        output_manager.error('Usage: %s post %s\n%s', BASENAME, st_post_options, st_post_help)
                        return
                else:
                    result = swift.post(container=container)
            if not result['success']:
                raise result['error']
        except SwiftError as e:
            output_manager.error(e.value)