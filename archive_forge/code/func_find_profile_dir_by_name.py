import os
import shutil
import errno
from pathlib import Path
from traitlets.config.configurable import LoggingConfigurable
from ..paths import get_ipython_package_dir
from ..utils.path import expand_path, ensure_dir_exists
from traitlets import Unicode, Bool, observe
@classmethod
def find_profile_dir_by_name(cls, ipython_dir, name=u'default', config=None):
    """Find an existing profile dir by profile name, return its ProfileDir.

        This searches through a sequence of paths for a profile dir.  If it
        is not found, a :class:`ProfileDirError` exception will be raised.

        The search path algorithm is:
        1. ``os.getcwd()`` # removed for security reason.
        2. ``ipython_dir``

        Parameters
        ----------
        ipython_dir : unicode or str
            The IPython directory to use.
        name : unicode or str
            The name of the profile.  The name of the profile directory
            will be "profile_<profile>".
        """
    dirname = u'profile_' + name
    paths = [ipython_dir]
    for p in paths:
        profile_dir = os.path.join(p, dirname)
        if os.path.isdir(profile_dir):
            return cls(location=profile_dir, config=config)
    else:
        raise ProfileDirError('Profile directory not found in paths: %s' % dirname)