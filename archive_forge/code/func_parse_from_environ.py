from __future__ import annotations
import typing as t
from io import BytesIO
from urllib.parse import parse_qsl
from ._internal import _plain_int
from .datastructures import FileStorage
from .datastructures import Headers
from .datastructures import MultiDict
from .exceptions import RequestEntityTooLarge
from .http import parse_options_header
from .sansio.multipart import Data
from .sansio.multipart import Epilogue
from .sansio.multipart import Field
from .sansio.multipart import File
from .sansio.multipart import MultipartDecoder
from .sansio.multipart import NeedData
from .wsgi import get_content_length
from .wsgi import get_input_stream
def parse_from_environ(self, environ: WSGIEnvironment) -> t_parse_result:
    """Parses the information from the environment as form data.

        :param environ: the WSGI environment to be used for parsing.
        :return: A tuple in the form ``(stream, form, files)``.
        """
    stream = get_input_stream(environ, max_content_length=self.max_content_length)
    content_length = get_content_length(environ)
    mimetype, options = parse_options_header(environ.get('CONTENT_TYPE'))
    return self.parse(stream, content_length=content_length, mimetype=mimetype, options=options)