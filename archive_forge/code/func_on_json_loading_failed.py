from __future__ import annotations
import functools
import json
import typing as t
from io import BytesIO
from .._internal import _wsgi_decoding_dance
from ..datastructures import CombinedMultiDict
from ..datastructures import EnvironHeaders
from ..datastructures import FileStorage
from ..datastructures import ImmutableMultiDict
from ..datastructures import iter_multi_items
from ..datastructures import MultiDict
from ..exceptions import BadRequest
from ..exceptions import UnsupportedMediaType
from ..formparser import default_stream_factory
from ..formparser import FormDataParser
from ..sansio.request import Request as _SansIORequest
from ..utils import cached_property
from ..utils import environ_property
from ..wsgi import _get_server
from ..wsgi import get_input_stream
def on_json_loading_failed(self, e: ValueError | None) -> t.Any:
    """Called if :meth:`get_json` fails and isn't silenced.

        If this method returns a value, it is used as the return value
        for :meth:`get_json`. The default implementation raises
        :exc:`~werkzeug.exceptions.BadRequest`.

        :param e: If parsing failed, this is the exception. It will be
            ``None`` if the content type wasn't ``application/json``.

        .. versionchanged:: 2.3
            Raise a 415 error instead of 400.
        """
    if e is not None:
        raise BadRequest(f'Failed to decode JSON object: {e}')
    raise UnsupportedMediaType("Did not attempt to load JSON data because the request Content-Type was not 'application/json'.")