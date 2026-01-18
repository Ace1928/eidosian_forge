import asyncio
import json
import os
import threading
import warnings
from enum import Enum
from functools import partial
from typing import Awaitable, Dict
import websockets
from jupyterlab_server.themes_handler import ThemesHandler
from markupsafe import Markup
from nbconvert.exporters.html import find_lab_theme
from .static_file_handler import TemplateStaticFileHandler
def include_lab_theme(base_url: str, name: str) -> str:
    theme_name, _ = find_lab_theme(name)
    settings = {'static_url_prefix': f'{base_url}voila/themes/', 'static_path': None}
    url = ThemesHandler.make_static_url(settings, f'{theme_name}/index.css')
    code = f'<link rel="stylesheet" type="text/css" href="{url}">'
    return Markup(code)