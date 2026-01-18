import argparse
import collections
import datetime
import functools
import os
import sys
import time
import uuid
from oslo_utils import encodeutils
import prettytable
from glance.common import exception
import glance.image_cache.client
from glance.version import version_info as version
def lookup_command(command_name):
    try:
        command = CACHE_COMMANDS[command_name]
        return command[0]
    except KeyError:
        print('\nError: "%s" is not a valid command.\n' % command_name)
        print(_format_command_help())
        sys.exit('Unknown command: %(cmd_name)s' % {'cmd_name': command_name})