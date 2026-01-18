from __future__ import annotations
import json as _json
import typing as t
from ..globals import current_app
from .provider import _default
def jsonify(*args: t.Any, **kwargs: t.Any) -> Response:
    """Serialize the given arguments as JSON, and return a
    :class:`~flask.Response` object with the ``application/json``
    mimetype. A dict or list returned from a view will be converted to a
    JSON response automatically without needing to call this.

    This requires an active request or application context, and calls
    :meth:`app.json.response() <flask.json.provider.JSONProvider.response>`.

    In debug mode, the output is formatted with indentation to make it
    easier to read. This may also be controlled by the provider.

    Either positional or keyword arguments can be given, not both.
    If no arguments are given, ``None`` is serialized.

    :param args: A single value to serialize, or multiple values to
        treat as a list to serialize.
    :param kwargs: Treat as a dict to serialize.

    .. versionchanged:: 2.2
        Calls ``current_app.json.response``, allowing an app to override
        the behavior.

    .. versionchanged:: 2.0.2
        :class:`decimal.Decimal` is supported by converting to a string.

    .. versionchanged:: 0.11
        Added support for serializing top-level arrays. This was a
        security risk in ancient browsers. See :ref:`security-json`.

    .. versionadded:: 0.2
    """
    return current_app.json.response(*args, **kwargs)