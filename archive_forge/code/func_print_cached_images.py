import datetime
import errno
import functools
import hashlib
import json
import os
import re
import sys
import threading
import urllib.parse
import uuid
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
import wrapt
from glanceclient._i18n import _
from glanceclient import exc
def print_cached_images(cached_images):
    cache_pt = prettytable.PrettyTable(('ID', 'State', 'Last Accessed (UTC)', 'Last Modified (UTC)', 'Size', 'Hits'))
    for item in cached_images:
        state = 'queued'
        last_accessed = 'N/A'
        last_modified = 'N/A'
        size = 'N/A'
        hits = 'N/A'
        if item == 'cached_images':
            state = 'cached'
            for image in cached_images[item]:
                last_accessed = image['last_accessed']
                if last_accessed == 0:
                    last_accessed = 'N/A'
                else:
                    last_accessed = datetime.datetime.utcfromtimestamp(last_accessed).isoformat()
                cache_pt.add_row((image['image_id'], state, last_accessed, datetime.datetime.utcfromtimestamp(image['last_modified']).isoformat(), image['size'], image['hits']))
        else:
            for image in cached_images[item]:
                cache_pt.add_row((image, state, last_accessed, last_modified, size, hits))
    print(cache_pt.get_string())