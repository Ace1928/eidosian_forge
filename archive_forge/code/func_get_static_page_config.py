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
def get_static_page_config(app_settings_dir: str | None=None, logger: Logger | None=None, level: str='all') -> dict[str, Any]:
    """Get the static page config for JupyterLab

    Parameters
    ----------
    logger: logger, optional
        An optional logging object
    level: string, optional ['all']
        The level at which to get config: can be 'all', 'user', 'sys_prefix', or 'system'
    """
    cm = _get_config_manager(level)
    return cm.get('page_config')