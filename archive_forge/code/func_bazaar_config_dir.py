import os
import sys
from .lazy_import import lazy_import
from breezy import (
from . import errors
def bazaar_config_dir():
    """Return per-user configuration directory as unicode string

    By default this is %APPDATA%/bazaar/2.0 on Windows, ~/.bazaar on Mac OS X
    and Linux.  On Mac OS X and Linux, if there is a $XDG_CONFIG_HOME/bazaar
    directory, that will be used instead

    TODO: Global option --config-dir to override this.
    """
    base = os.environ.get('BZR_HOME')
    if sys.platform == 'win32':
        if base is None:
            base = win32utils.get_appdata_location()
        if base is None:
            base = win32utils.get_home_location()
        return osutils.pathjoin(base, 'bazaar', '2.0')
    if base is None:
        xdg_dir = os.environ.get('XDG_CONFIG_HOME')
        if xdg_dir is None:
            xdg_dir = osutils.pathjoin(osutils._get_home_dir(), '.config')
        xdg_dir = osutils.pathjoin(xdg_dir, 'bazaar')
        if osutils.isdir(xdg_dir):
            trace.mutter('Using configuration in XDG directory %s.' % xdg_dir)
            return xdg_dir
        base = osutils._get_home_dir()
    return osutils.pathjoin(base, '.bazaar')