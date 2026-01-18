import functools
import os
import site
import sys
import sysconfig
import typing
from pip._internal.exceptions import InstallationError
from pip._internal.utils import appdirs
from pip._internal.utils.virtualenv import running_under_virtualenv
def get_src_prefix() -> str:
    if running_under_virtualenv():
        src_prefix = os.path.join(sys.prefix, 'src')
    else:
        try:
            src_prefix = os.path.join(os.getcwd(), 'src')
        except OSError:
            sys.exit('The folder you are executing pip from can no longer be found.')
    return os.path.abspath(src_prefix)