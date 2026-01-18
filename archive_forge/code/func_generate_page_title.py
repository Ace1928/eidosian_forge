import os
from tornado import web
from jupyter_server.utils import url_path_join, url_escape
from nbclient.util import ensure_async
from .utils import get_server_root_dir
from .handler import BaseVoilaHandler
def generate_page_title(self, path):
    parts = path.split('/')
    if len(parts) > 3:
        parts = parts[-2:]
    page_title = url_path_join(*parts)
    if page_title:
        return page_title + '/'
    else:
        return 'Voilà Home'