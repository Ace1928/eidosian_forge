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
@catch_error('show cached images')
def list_cached(args):
    """%(prog)s list-cached [options]

    List all images currently cached.
    """
    client = get_client(args)
    images = client.get_cached_images()
    if not images:
        print('No cached images.')
        return SUCCESS
    print('Found %d cached images...' % len(images))
    pretty_table = prettytable.PrettyTable(('ID', 'Last Accessed (UTC)', 'Last Modified (UTC)', 'Size', 'Hits'))
    pretty_table.align['Size'] = 'r'
    pretty_table.align['Hits'] = 'r'
    for image in images:
        last_accessed = image['last_accessed']
        if last_accessed == 0:
            last_accessed = 'N/A'
        else:
            last_accessed = datetime.datetime.utcfromtimestamp(last_accessed).isoformat()
        pretty_table.add_row((image['image_id'], last_accessed, datetime.datetime.utcfromtimestamp(image['last_modified']).isoformat(), image['size'], image['hits']))
    print(pretty_table.get_string())
    return SUCCESS