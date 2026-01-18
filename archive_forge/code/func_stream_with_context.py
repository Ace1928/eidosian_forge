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
def stream_with_context(generator_or_function: t.Iterator[t.AnyStr] | t.Callable[..., t.Iterator[t.AnyStr]]) -> t.Iterator[t.AnyStr]:
    """Request contexts disappear when the response is started on the server.
    This is done for efficiency reasons and to make it less likely to encounter
    memory leaks with badly written WSGI middlewares.  The downside is that if
    you are using streamed responses, the generator cannot access request bound
    information any more.

    This function however can help you keep the context around for longer::

        from flask import stream_with_context, request, Response

        @app.route('/stream')
        def streamed_response():
            @stream_with_context
            def generate():
                yield 'Hello '
                yield request.args['name']
                yield '!'
            return Response(generate())

    Alternatively it can also be used around a specific generator::

        from flask import stream_with_context, request, Response

        @app.route('/stream')
        def streamed_response():
            def generate():
                yield 'Hello '
                yield request.args['name']
                yield '!'
            return Response(stream_with_context(generate()))

    .. versionadded:: 0.9
    """
    try:
        gen = iter(generator_or_function)
    except TypeError:

        def decorator(*args: t.Any, **kwargs: t.Any) -> t.Any:
            gen = generator_or_function(*args, **kwargs)
            return stream_with_context(gen)
        return update_wrapper(decorator, generator_or_function)

    def generator() -> t.Iterator[t.AnyStr | None]:
        ctx = _cv_request.get(None)
        if ctx is None:
            raise RuntimeError("'stream_with_context' can only be used when a request context is active, such as in a view function.")
        with ctx:
            yield None
            try:
                yield from gen
            finally:
                if hasattr(gen, 'close'):
                    gen.close()
    wrapped_g = generator()
    next(wrapped_g)
    return wrapped_g