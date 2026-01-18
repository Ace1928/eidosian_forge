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
def add_thin_pack(self, read_all, read_some):
    """Read a thin pack.

        Read it from a stream and complete it in a temporary file.
        Then the pack and the corresponding index file are uploaded to Swift.
        """
    fd, path = tempfile.mkstemp(prefix='tmp_pack_')
    f = os.fdopen(fd, 'w+b')
    try:
        indexer = PackIndexer(f, resolve_ext_ref=self.get_raw)
        copier = PackStreamCopier(read_all, read_some, f, delta_iter=indexer)
        copier.verify()
        return self._complete_thin_pack(f, path, copier, indexer)
    finally:
        f.close()
        os.unlink(path)