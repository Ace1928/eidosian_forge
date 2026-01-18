import re
import warnings
from typing import Optional, no_type_check
from urllib.parse import urlparse
from tornado import ioloop, web
from tornado.iostream import IOStream
from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.utils import JupyterServerAuthWarning
def clear_cookie(self, *args, **kwargs):
    """meaningless for websockets"""