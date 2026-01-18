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
def swift_auth_v1(self):
    self.user = self.user.replace(';', ':')
    auth_httpclient = HTTPClient.from_url(self.auth_url, connection_timeout=self.http_timeout, network_timeout=self.http_timeout)
    headers = {'X-Auth-User': self.user, 'X-Auth-Key': self.password}
    path = urlparse.urlparse(self.auth_url).path
    ret = auth_httpclient.request('GET', path, headers=headers)
    if ret.status_code < 200 or ret.status_code >= 300:
        raise SwiftException('AUTH v1.0 request failed on ' + '{} with error code {} ({})'.format(str(auth_httpclient.get_base_url()) + path, ret.status_code, str(ret.items())))
    storage_url = ret['X-Storage-Url']
    token = ret['X-Auth-Token']
    return (storage_url, token)