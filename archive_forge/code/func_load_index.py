import os.path as _path
import csv as _csv
from netaddr.compat import _open_binary
from netaddr.core import Subscriber, Publisher
def load_index(index, fp):
    """Load index from file into index data structure."""
    try:
        for row in _csv.reader([x.decode('UTF-8') for x in fp]):
            key, offset, size = [int(_) for _ in row]
            index.setdefault(key, [])
            index[key].append((offset, size))
    finally:
        fp.close()