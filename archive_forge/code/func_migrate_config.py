from __future__ import annotations
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from traitlets.config.loader import JSONFileConfigLoader, PyFileConfigLoader
from traitlets.log import get_logger
from .application import JupyterApp
from .paths import jupyter_config_dir, jupyter_data_dir
from .utils import ensure_dir_exists
def migrate_config(name: str, env: Any) -> list[Any]:
    """Migrate a config file.

    Includes substitutions for updated configurable names.
    """
    log = get_logger()
    src_base = str(Path(f'{env['profile']}', f'ipython_{name}_config'))
    dst_base = str(Path(f'{env['jupyter_config']}', f'jupyter_{name}_config'))
    loaders = {'.py': PyFileConfigLoader, '.json': JSONFileConfigLoader}
    migrated = []
    for ext in ('.py', '.json'):
        src = src_base + ext
        dst = dst_base + ext
        if Path(src).exists():
            cfg = loaders[ext](src).load_config()
            if cfg:
                if migrate_file(src, dst, substitutions=config_substitutions):
                    migrated.append(src)
            else:
                log.debug('Not migrating empty config file: %s', src)
    return migrated