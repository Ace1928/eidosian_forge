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
def get_flashed_messages(with_categories: bool=False, category_filter: t.Iterable[str]=()) -> list[str] | list[tuple[str, str]]:
    """Pulls all flashed messages from the session and returns them.
    Further calls in the same request to the function will return
    the same messages.  By default just the messages are returned,
    but when `with_categories` is set to ``True``, the return value will
    be a list of tuples in the form ``(category, message)`` instead.

    Filter the flashed messages to one or more categories by providing those
    categories in `category_filter`.  This allows rendering categories in
    separate html blocks.  The `with_categories` and `category_filter`
    arguments are distinct:

    * `with_categories` controls whether categories are returned with message
      text (``True`` gives a tuple, where ``False`` gives just the message text).
    * `category_filter` filters the messages down to only those matching the
      provided categories.

    See :doc:`/patterns/flashing` for examples.

    .. versionchanged:: 0.3
       `with_categories` parameter added.

    .. versionchanged:: 0.9
        `category_filter` parameter added.

    :param with_categories: set to ``True`` to also receive categories.
    :param category_filter: filter of categories to limit return values.  Only
                            categories in the list will be returned.
    """
    flashes = request_ctx.flashes
    if flashes is None:
        flashes = session.pop('_flashes') if '_flashes' in session else []
        request_ctx.flashes = flashes
    if category_filter:
        flashes = list(filter(lambda f: f[0] in category_filter, flashes))
    if not with_categories:
        return [x[1] for x in flashes]
    return flashes