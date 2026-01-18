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
def remove_if_equals(self, name, old_ref):
    """Remove a refname only if it currently equals old_ref."""
    if name == 'HEAD':
        return True
    refs = self._load_check_ref(name, old_ref)
    if not isinstance(refs, dict):
        return False
    del refs[name]
    self._write_refs(refs)
    del self._refs[name]
    return True