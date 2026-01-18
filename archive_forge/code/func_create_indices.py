import os.path as _path
import csv as _csv
from netaddr.compat import _open_binary
from netaddr.core import Subscriber, Publisher
def create_indices():
    """Create indices for OUI and IAB file based lookups"""
    create_index_from_registry(_path.join(_path.dirname(__file__), 'oui.txt'), _path.join(_path.dirname(__file__), 'oui.idx'), OUIIndexParser)
    create_index_from_registry(_path.join(_path.dirname(__file__), 'iab.txt'), _path.join(_path.dirname(__file__), 'iab.idx'), IABIndexParser)