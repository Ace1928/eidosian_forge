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
def st_list(parser, args, output_manager, return_parser=False):

    def _print_stats(options, stats, human, totals):
        container = stats.get('container', None)
        for item in stats['listing']:
            item_name = item.get('name')
            if not options['long'] and (not human) and (not options['versions']):
                output_manager.print_msg(item.get('name', item.get('subdir')))
            else:
                if not container:
                    item_bytes = item.get('bytes')
                    byte_str = prt_bytes(item_bytes, human)
                    count = item.get('count')
                    totals['count'] += count
                    try:
                        meta = item.get('meta')
                        utc = gmtime(float(meta.get('x-timestamp')))
                        datestamp = strftime('%Y-%m-%d %H:%M:%S', utc)
                    except TypeError:
                        datestamp = '????-??-?? ??:??:??'
                    storage_policy = meta.get('x-storage-policy', '???')
                    if not options['totals']:
                        output_manager.print_msg('%12s %s %s %-15s %s', count, byte_str, datestamp, storage_policy, item_name)
                else:
                    subdir = item.get('subdir')
                    content_type = item.get('content_type')
                    if subdir is None:
                        item_bytes = item.get('bytes')
                        byte_str = prt_bytes(item_bytes, human)
                        date, xtime = item.get('last_modified').split('T')
                        xtime = xtime.split('.')[0]
                    else:
                        item_bytes = 0
                        byte_str = prt_bytes(item_bytes, human)
                        date = xtime = ''
                        item_name = subdir
                    if not options['totals']:
                        if options['versions']:
                            output_manager.print_msg('%s %10s %8s %16s %24s %s', byte_str, date, xtime, item.get('version_id', 'null'), content_type, item_name)
                        else:
                            output_manager.print_msg('%s %10s %8s %24s %s', byte_str, date, xtime, content_type, item_name)
                totals['bytes'] += item_bytes
    parser.add_argument('-l', '--long', dest='long', action='store_true', default=False, help='Long listing format, similar to ls -l.')
    parser.add_argument('--lh', dest='human', action='store_true', default=False, help='Report sizes in human readable format, similar to ls -lh.')
    parser.add_argument('-t', '--totals', dest='totals', help='used with -l or --lh, only report totals.', action='store_true', default=False)
    parser.add_argument('-p', '--prefix', dest='prefix', help='Only list items beginning with the prefix.')
    parser.add_argument('-d', '--delimiter', dest='delimiter', help='Roll up items with the given delimiter. For containers only. See OpenStack Swift API documentation for what this means.')
    parser.add_argument('-j', '--json', action='store_true', help='print listing information in json')
    parser.add_argument('--versions', action='store_true', help='display all versions')
    parser.add_argument('-H', '--header', action='append', dest='header', default=[], help='Adds a custom request header to use for listing.')
    if return_parser:
        return parser
    options, args = parse_args(parser, args)
    args = args[1:]
    if options['delimiter'] and (not args):
        exit('-d option only allowed for container listings')
    if options['versions'] and (not args):
        exit('--versions option only allowed for container listings')
    human = options.pop('human')
    if human:
        options['long'] = True
    if options['totals'] and (not options['long']):
        output_manager.error('Listing totals only works with -l or --lh.')
        return
    with SwiftService(options=options) as swift:
        try:
            if not args:
                stats_parts_gen = swift.list()
                container = None
            else:
                container = args[0]
                args = args[1:]
                if '/' in container or args:
                    output_manager.error('Usage: %s list %s\n%s', BASENAME, st_list_options, st_list_help)
                    return
                else:
                    stats_parts_gen = swift.list(container=container)
            if options.get('json', False):

                def listing(stats_parts_gen=stats_parts_gen):
                    for stats in stats_parts_gen:
                        if stats['success']:
                            for item in stats['listing']:
                                yield item
                        else:
                            raise stats['error']
                json.dump(JSONableIterable(listing()), output_manager.print_stream, sort_keys=True, indent=2)
                output_manager.print_msg('')
                return
            totals = {'count': 0, 'bytes': 0}
            for stats in stats_parts_gen:
                if stats['success']:
                    _print_stats(options, stats, human, totals)
                else:
                    raise stats['error']
            if options['long'] or human:
                if container is None:
                    output_manager.print_msg('%12s %s', prt_bytes(totals['count'], True), prt_bytes(totals['bytes'], human))
                else:
                    output_manager.print_msg(prt_bytes(totals['bytes'], human))
        except SwiftError as e:
            output_manager.error(e.value)