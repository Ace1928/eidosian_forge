import sys
import os
import builtins
import _sitebuiltins
import io
import stat
def getusersitepackages():
    """Returns the user-specific site-packages directory path.

    If the global variable ``USER_SITE`` is not initialized yet, this
    function will also set it.
    """
    global USER_SITE, ENABLE_USER_SITE
    userbase = getuserbase()
    if USER_SITE is None:
        if userbase is None:
            ENABLE_USER_SITE = False
        else:
            USER_SITE = _get_path(userbase)
    return USER_SITE