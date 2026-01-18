import os.path as _path
import csv as _csv
from netaddr.compat import _open_binary
from netaddr.core import Subscriber, Publisher
def create_index_from_registry(registry_fh, index_path, parser):
    """Generate an index files from the IEEE registry file."""
    oui_parser = parser(registry_fh)
    oui_parser.attach(FileIndexer(index_path))
    oui_parser.parse()