import os
import shutil
import errno
from pathlib import Path
from traitlets.config.configurable import LoggingConfigurable
from ..paths import get_ipython_package_dir
from ..utils.path import expand_path, ensure_dir_exists
from traitlets import Unicode, Bool, observe
@classmethod
def find_profile_dir(cls, profile_dir, config=None):
    """Find/create a profile dir and return its ProfileDir.

        This will create the profile directory if it doesn't exist.

        Parameters
        ----------
        profile_dir : unicode or str
            The path of the profile directory.
        """
    profile_dir = expand_path(profile_dir)
    if not os.path.isdir(profile_dir):
        raise ProfileDirError('Profile directory not found: %s' % profile_dir)
    return cls(location=profile_dir, config=config)