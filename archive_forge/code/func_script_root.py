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
@property
def script_root(self) -> str:
    """Alias for :attr:`self.root_path`. ``environ["SCRIPT_ROOT"]``
        without a trailing slash.
        """
    return self.root_path