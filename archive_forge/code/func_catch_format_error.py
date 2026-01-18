import abc
import sys
import traceback
import warnings
from io import StringIO
from decorator import decorator
from traitlets.config.configurable import Configurable
from .getipython import get_ipython
from ..utils.sentinel import Sentinel
from ..utils.dir2 import get_real_method
from ..lib import pretty
from traitlets import (
from typing import Any
@decorator
def catch_format_error(method, self, *args, **kwargs):
    """show traceback on failed format call"""
    try:
        r = method(self, *args, **kwargs)
    except NotImplementedError:
        return self._check_return(None, args[0])
    except Exception:
        exc_info = sys.exc_info()
        ip = get_ipython()
        if ip is not None:
            ip.showtraceback(exc_info)
        else:
            traceback.print_exception(*exc_info)
        return self._check_return(None, args[0])
    return self._check_return(r, args[0])