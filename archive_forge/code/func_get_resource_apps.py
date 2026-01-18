import argparse
import functools
import gzip
import io
import mimetypes
import posixpath
import zipfile
from werkzeug import utils
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard import version
def get_resource_apps(self):
    apps = {}
    if not self._assets_zip_provider:
        return apps
    with self._assets_zip_provider() as fp:
        with zipfile.ZipFile(fp) as zip_:
            for path in zip_.namelist():
                content = zip_.read(path)
                if path == 'index.html':
                    apps['/' + path] = functools.partial(self._serve_index, content)
                    continue
                gzipped_asset_bytes = _gzip(content)
                wsgi_app = functools.partial(self._serve_asset, path, gzipped_asset_bytes)
                apps['/' + path] = wsgi_app
    apps['/'] = apps['/index.html']
    return apps