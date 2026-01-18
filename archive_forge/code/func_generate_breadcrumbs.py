import os
from tornado import web
from jupyter_server.utils import url_path_join, url_escape
from nbclient.util import ensure_async
from .utils import get_server_root_dir
from .handler import BaseVoilaHandler
def generate_breadcrumbs(self, path):
    breadcrumbs = [(url_path_join(self.base_url, 'voila/tree'), '')]
    parts = path.split('/')
    for i in range(len(parts)):
        if parts[i]:
            link = url_path_join(self.base_url, 'voila/tree', url_escape(url_path_join(*parts[:i + 1])))
            breadcrumbs.append((link, parts[i]))
    return breadcrumbs