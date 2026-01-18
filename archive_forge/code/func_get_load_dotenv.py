from __future__ import annotations
import importlib.util
import os
import sys
import typing as t
from datetime import datetime
from functools import lru_cache
from functools import update_wrapper
import werkzeug.utils
from werkzeug.exceptions import abort as _wz_abort
from werkzeug.utils import redirect as _wz_redirect
from werkzeug.wrappers import Response as BaseResponse
from .globals import _cv_request
from .globals import current_app
from .globals import request
from .globals import request_ctx
from .globals import session
from .signals import message_flashed
def get_load_dotenv(default: bool=True) -> bool:
    """Get whether the user has disabled loading default dotenv files by
    setting :envvar:`FLASK_SKIP_DOTENV`. The default is ``True``, load
    the files.

    :param default: What to return if the env var isn't set.
    """
    val = os.environ.get('FLASK_SKIP_DOTENV')
    if not val:
        return default
    return val.lower() in ('0', 'false', 'no')