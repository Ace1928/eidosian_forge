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
def swift_auth_v2(self):
    self.tenant, self.user = self.user.split(';')
    auth_dict = {}
    auth_dict['auth'] = {'passwordCredentials': {'username': self.user, 'password': self.password}, 'tenantName': self.tenant}
    auth_json = json.dumps(auth_dict)
    headers = {'Content-Type': 'application/json'}
    auth_httpclient = HTTPClient.from_url(self.auth_url, connection_timeout=self.http_timeout, network_timeout=self.http_timeout)
    path = urlparse.urlparse(self.auth_url).path
    if not path.endswith('tokens'):
        path = posixpath.join(path, 'tokens')
    ret = auth_httpclient.request('POST', path, body=auth_json, headers=headers)
    if ret.status_code < 200 or ret.status_code >= 300:
        raise SwiftException('AUTH v2.0 request failed on ' + '{} with error code {} ({})'.format(str(auth_httpclient.get_base_url()) + path, ret.status_code, str(ret.items())))
    auth_ret_json = json.loads(ret.read())
    token = auth_ret_json['access']['token']['id']
    catalogs = auth_ret_json['access']['serviceCatalog']
    object_store = next((o_store for o_store in catalogs if o_store['type'] == 'object-store'))
    endpoints = object_store['endpoints']
    endpoint = next((endp for endp in endpoints if endp['region'] == self.region_name))
    return (endpoint[self.endpoint_type], token)