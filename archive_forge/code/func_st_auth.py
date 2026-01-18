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
def st_auth(parser, args, thread_manager, return_parser=False):
    if return_parser:
        return parser
    options, args = parse_args(parser, args)
    if options['verbose'] > 1:
        if options['auth_version'] in ('1', '1.0'):
            print('export ST_AUTH=%s' % sh_quote(options['auth']))
            print('export ST_USER=%s' % sh_quote(options['user']))
            print('export ST_KEY=%s' % sh_quote(options['key']))
        else:
            print('export OS_IDENTITY_API_VERSION=%s' % sh_quote(options['auth_version']))
            print('export OS_AUTH_VERSION=%s' % sh_quote(options['auth_version']))
            print('export OS_AUTH_URL=%s' % sh_quote(options['auth']))
            for k, v in sorted(options.items()):
                if v and k.startswith('os_') and (k not in ('os_auth_url', 'os_options')):
                    print('export %s=%s' % (k.upper(), sh_quote(v)))
    else:
        conn = get_conn(options)
        url, token = conn.get_auth()
        print('export OS_STORAGE_URL=%s' % sh_quote(url))
        print('export OS_AUTH_TOKEN=%s' % sh_quote(token))