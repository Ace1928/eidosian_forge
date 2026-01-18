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
def get_object_at(self, offset):
    if offset in self._offset_cache:
        return self._offset_cache[offset]
    assert offset >= self._header_size
    pack_reader = SwiftPackReader(self.scon, self._filename, self.pack_length)
    pack_reader.seek(offset)
    unpacked, _ = unpack_object(pack_reader.read)
    return (unpacked.pack_type_num, unpacked._obj())