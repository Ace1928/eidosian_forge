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
def cmd_daemon(args):
    """Entry point for starting a TCP git server."""
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-l', '--listen_address', dest='listen_address', default='127.0.0.1', help='Binding IP address.')
    parser.add_option('-p', '--port', dest='port', type=int, default=TCP_GIT_PORT, help='Binding TCP port.')
    parser.add_option('-c', '--swift_config', dest='swift_config', default='', help='Path to the configuration file for Swift backend.')
    options, args = parser.parse_args(args)
    try:
        import gevent
        import geventhttpclient
    except ImportError:
        print('gevent and geventhttpclient libraries are mandatory  for use the Swift backend.')
        sys.exit(1)
    import gevent.monkey
    gevent.monkey.patch_socket()
    from dulwich import log_utils
    logger = log_utils.getLogger(__name__)
    conf = load_conf(options.swift_config)
    backend = SwiftSystemBackend(logger, conf)
    log_utils.default_logging_config()
    server = TCPGitServer(backend, options.listen_address, port=options.port)
    server.serve_forever()