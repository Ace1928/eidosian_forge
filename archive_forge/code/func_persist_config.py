import getpass
import hashlib
import json
import os
import random
import traceback
import warnings
from contextlib import contextmanager
from jupyter_core.paths import jupyter_config_dir
from traitlets.config import Config
from traitlets.config.loader import ConfigFileNotFound, JSONFileConfigLoader
@contextmanager
def persist_config(config_file=None, mode=384):
    """Context manager that can be used to modify a config object

    On exit of the context manager, the config will be written back to disk,
    by default with user-only (600) permissions.
    """
    if config_file is None:
        config_file = os.path.join(jupyter_config_dir(), 'jupyter_server_config.json')
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    loader = JSONFileConfigLoader(os.path.basename(config_file), os.path.dirname(config_file))
    try:
        config = loader.load_config()
    except ConfigFileNotFound:
        config = Config()
    yield config
    with open(config_file, 'w', encoding='utf8') as f:
        f.write(json.dumps(config, indent=2))
    try:
        os.chmod(config_file, mode)
    except Exception:
        tb = traceback.format_exc()
        warnings.warn(f'Failed to set permissions on {config_file}:\n{tb}', RuntimeWarning, stacklevel=2)