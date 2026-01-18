from __future__ import annotations
import errno
import json
import os
import types
import typing as t
from werkzeug.utils import import_string
def from_prefixed_env(self, prefix: str='FLASK', *, loads: t.Callable[[str], t.Any]=json.loads) -> bool:
    """Load any environment variables that start with ``FLASK_``,
        dropping the prefix from the env key for the config key. Values
        are passed through a loading function to attempt to convert them
        to more specific types than strings.

        Keys are loaded in :func:`sorted` order.

        The default loading function attempts to parse values as any
        valid JSON type, including dicts and lists.

        Specific items in nested dicts can be set by separating the
        keys with double underscores (``__``). If an intermediate key
        doesn't exist, it will be initialized to an empty dict.

        :param prefix: Load env vars that start with this prefix,
            separated with an underscore (``_``).
        :param loads: Pass each string value to this function and use
            the returned value as the config value. If any error is
            raised it is ignored and the value remains a string. The
            default is :func:`json.loads`.

        .. versionadded:: 2.1
        """
    prefix = f'{prefix}_'
    len_prefix = len(prefix)
    for key in sorted(os.environ):
        if not key.startswith(prefix):
            continue
        value = os.environ[key]
        try:
            value = loads(value)
        except Exception:
            pass
        key = key[len_prefix:]
        if '__' not in key:
            self[key] = value
            continue
        current = self
        *parts, tail = key.split('__')
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[tail] = value
    return True