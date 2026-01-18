import asyncio
import os
from typing import Dict
from pathlib import Path
import tornado.web
from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.utils import url_path_join
from nbclient.util import ensure_async
from tornado.httputil import split_host_and_port
from traitlets.traitlets import Bool
from ._version import __version__
from .notebook_renderer import NotebookRenderer
from .request_info_handler import RequestInfoSocketHandler
from .utils import ENV_VARIABLE, create_include_assets_functions
def should_use_rendered_notebook(self, notebook_data: Dict, pool_size: int, template_name: str, theme: str, request_args: Dict) -> Bool:
    if pool_size == 0:
        return False
    if len(notebook_data) == 0:
        return False
    rendered_template = notebook_data.get('template')
    rendered_theme = notebook_data.get('theme')
    if template_name is not None and template_name != rendered_template:
        return False
    if theme is not None and rendered_theme != theme:
        return False
    return True