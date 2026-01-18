import os
import shutil
import errno
from pathlib import Path
from traitlets.config.configurable import LoggingConfigurable
from ..paths import get_ipython_package_dir
from ..utils.path import expand_path, ensure_dir_exists
from traitlets import Unicode, Bool, observe
@classmethod
def create_profile_dir(cls, profile_dir, config=None):
    """Create a new profile directory given a full path.

        Parameters
        ----------
        profile_dir : str
            The full path to the profile directory.  If it does exist, it will
            be used.  If not, it will be created.
        """
    return cls(location=profile_dir, config=config)