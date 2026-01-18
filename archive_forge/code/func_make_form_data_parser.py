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
def make_form_data_parser(self) -> FormDataParser:
    """Creates the form data parser. Instantiates the
        :attr:`form_data_parser_class` with some parameters.

        .. versionadded:: 0.8
        """
    return self.form_data_parser_class(stream_factory=self._get_file_stream, max_form_memory_size=self.max_form_memory_size, max_content_length=self.max_content_length, max_form_parts=self.max_form_parts, cls=self.parameter_storage_class)