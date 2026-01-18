from __future__ import annotations
import json
import os.path as osp
from glob import iglob
from itertools import chain
from logging import Logger
from os.path import join as pjoin
from typing import Any
import json5
from jupyter_core.paths import SYSTEM_CONFIG_PATH, jupyter_config_dir, jupyter_path
from jupyter_server.services.config.manager import ConfigManager, recursive_update
from jupyter_server.utils import url_path_join as ujoin
from traitlets import Bool, HasTraits, List, Unicode, default
def get_package_url(data: dict[str, Any]) -> str:
    """Get the url from the extension data"""
    if 'homepage' in data:
        url = data['homepage']
    elif 'repository' in data and isinstance(data['repository'], dict):
        url = data['repository'].get('url', '')
    else:
        url = ''
    return url