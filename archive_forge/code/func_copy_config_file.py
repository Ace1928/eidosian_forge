import os
import shutil
import errno
from pathlib import Path
from traitlets.config.configurable import LoggingConfigurable
from ..paths import get_ipython_package_dir
from ..utils.path import expand_path, ensure_dir_exists
from traitlets import Unicode, Bool, observe
def copy_config_file(self, config_file: str, path: Path, overwrite=False) -> bool:
    """Copy a default config file into the active profile directory.

        Default configuration files are kept in :mod:`IPython.core.profile`.
        This function moves these from that location to the working profile
        directory.
        """
    dst = Path(os.path.join(self.location, config_file))
    if dst.exists() and (not overwrite):
        return False
    if path is None:
        path = os.path.join(get_ipython_package_dir(), u'core', u'profile', u'default')
    assert isinstance(path, Path)
    src = path / config_file
    shutil.copy(src, dst)
    return True