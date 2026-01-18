import json
import os
import posixpath
import stat
import sys
import tempfile
import urllib.parse as urlparse
import zlib
from configparser import ConfigParser
from io import BytesIO
from geventhttpclient import HTTPClient
from ..greenthreads import GreenThreadsMissingObjectFinder
from ..lru_cache import LRUSizeCache
from ..object_store import INFODIR, PACKDIR, PackBasedObjectStore
from ..objects import S_ISGITLINK, Blob, Commit, Tag, Tree
from ..pack import (
from ..protocol import TCP_GIT_PORT
from ..refs import InfoRefsContainer, read_info_refs, write_info_refs
from ..repo import OBJECTDIR, BaseRepo
from ..server import Backend, TCPGitServer
def swift_load_pack_index(scon, filename):
    """Read a pack index file from Swift.

    Args:
      scon: a `SwiftConnector` instance
      filename: Path to the index file objectise
    Returns: a `PackIndexer` instance
    """
    with scon.get_object(filename) as f:
        return load_pack_index_file(filename, f)