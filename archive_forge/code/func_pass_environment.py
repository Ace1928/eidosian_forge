import enum
import json
import os
import re
import typing as t
from collections import abc
from collections import deque
from random import choice
from random import randrange
from threading import Lock
from types import CodeType
from urllib.parse import quote_from_bytes
import markupsafe
def pass_environment(f: F) -> F:
    """Pass the :class:`~jinja2.Environment` as the first argument to
    the decorated function when called while rendering a template.

    Can be used on functions, filters, and tests.

    .. versionadded:: 3.0.0
        Replaces ``environmentfunction`` and ``environmentfilter``.
    """
    f.jinja_pass_arg = _PassArg.environment
    return f