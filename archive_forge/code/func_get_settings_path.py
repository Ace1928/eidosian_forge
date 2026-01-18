import os
import sys
import zipfile
import weakref
from io import BytesIO
import pyglet
def get_settings_path(name):
    """Get a directory to save user preferences.

    Different platforms have different conventions for where to save user
    preferences, saved games, and settings.  This function implements those
    conventions.  Note that the returned path may not exist: applications
    should use ``os.makedirs`` to construct it if desired.

    On Linux, a directory `name` in the user's configuration directory is
    returned (usually under ``~/.config``).

    On Windows (including under Cygwin) the `name` directory in the user's
    ``Application Settings`` directory is returned.

    On Mac OS X the `name` directory under ``~/Library/Application Support``
    is returned.

    :Parameters:
        `name` : str
            The name of the application.

    :rtype: str
    """
    if pyglet.compat_platform in ('cygwin', 'win32'):
        if 'APPDATA' in os.environ:
            return os.path.join(os.environ['APPDATA'], name)
        else:
            return os.path.expanduser(f'~/{name}')
    elif pyglet.compat_platform == 'darwin':
        return os.path.expanduser(f'~/Library/Application Support/{name}')
    elif pyglet.compat_platform.startswith('linux'):
        if 'XDG_CONFIG_HOME' in os.environ:
            return os.path.join(os.environ['XDG_CONFIG_HOME'], name)
        else:
            return os.path.expanduser(f'~/.config/{name}')
    else:
        return os.path.expanduser(f'~/.{name}')