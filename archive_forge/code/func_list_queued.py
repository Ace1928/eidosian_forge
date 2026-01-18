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
@catch_error('show queued images')
def list_queued(args):
    """%(prog)s list-queued [options]

    List all images currently queued for caching.
    """
    client = get_client(args)
    images = client.get_queued_images()
    if not images:
        print('No queued images.')
        return SUCCESS
    print('Found %d queued images...' % len(images))
    pretty_table = prettytable.PrettyTable(('ID',))
    for image in images:
        pretty_table.add_row((image,))
    print(pretty_table.get_string())