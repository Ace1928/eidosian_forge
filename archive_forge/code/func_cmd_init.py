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
def cmd_init(args):
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-c', '--swift_config', dest='swift_config', default='', help='Path to the configuration file for Swift backend.')
    options, args = parser.parse_args(args)
    conf = load_conf(options.swift_config)
    if args == []:
        parser.error('missing repository name')
    repo = args[0]
    scon = SwiftConnector(repo, conf)
    SwiftRepo.init_bare(scon, conf)