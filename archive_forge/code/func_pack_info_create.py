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
def pack_info_create(pack_data, pack_index):
    pack = Pack.from_objects(pack_data, pack_index)
    info = {}
    for obj in pack.iterobjects():
        if obj.type_num == Commit.type_num:
            info[obj.id] = (obj.type_num, obj.parents, obj.tree)
        elif obj.type_num == Tree.type_num:
            shas = [(s, n, not stat.S_ISDIR(m)) for n, m, s in obj.items() if not S_ISGITLINK(m)]
            info[obj.id] = (obj.type_num, shas)
        elif obj.type_num == Blob.type_num:
            info[obj.id] = None
        elif obj.type_num == Tag.type_num:
            info[obj.id] = (obj.type_num, obj.object[1])
    return zlib.compress(json.dumps(info))