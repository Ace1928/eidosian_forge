import os
import shutil
import errno
from pathlib import Path
from traitlets.config.configurable import LoggingConfigurable
from ..paths import get_ipython_package_dir
from ..utils.path import expand_path, ensure_dir_exists
from traitlets import Unicode, Bool, observe
def check_dirs(self):
    self.check_security_dir()
    self.check_log_dir()
    self.check_pid_dir()
    self.check_startup_dir()