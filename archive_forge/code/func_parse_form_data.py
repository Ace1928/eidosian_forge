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
def parse_form_data(environ: WSGIEnvironment, stream_factory: TStreamFactory | None=None, max_form_memory_size: int | None=None, max_content_length: int | None=None, cls: type[MultiDict] | None=None, silent: bool=True, *, max_form_parts: int | None=None) -> t_parse_result:
    """Parse the form data in the environ and return it as tuple in the form
    ``(stream, form, files)``.  You should only call this method if the
    transport method is `POST`, `PUT`, or `PATCH`.

    If the mimetype of the data transmitted is `multipart/form-data` the
    files multidict will be filled with `FileStorage` objects.  If the
    mimetype is unknown the input stream is wrapped and returned as first
    argument, else the stream is empty.

    This is a shortcut for the common usage of :class:`FormDataParser`.

    :param environ: the WSGI environment to be used for parsing.
    :param stream_factory: An optional callable that returns a new read and
                           writeable file descriptor.  This callable works
                           the same as :meth:`Response._get_file_stream`.
    :param max_form_memory_size: the maximum number of bytes to be accepted for
                           in-memory stored form data.  If the data
                           exceeds the value specified an
                           :exc:`~exceptions.RequestEntityTooLarge`
                           exception is raised.
    :param max_content_length: If this is provided and the transmitted data
                               is longer than this value an
                               :exc:`~exceptions.RequestEntityTooLarge`
                               exception is raised.
    :param cls: an optional dict class to use.  If this is not specified
                       or `None` the default :class:`MultiDict` is used.
    :param silent: If set to False parsing errors will not be caught.
    :param max_form_parts: The maximum number of multipart parts to be parsed. If this
        is exceeded, a :exc:`~exceptions.RequestEntityTooLarge` exception is raised.
    :return: A tuple in the form ``(stream, form, files)``.

    .. versionchanged:: 3.0
        The ``charset`` and ``errors`` parameters were removed.

    .. versionchanged:: 2.3
        Added the ``max_form_parts`` parameter.

    .. versionadded:: 0.5.1
       Added the ``silent`` parameter.

    .. versionadded:: 0.5
       Added the ``max_form_memory_size``, ``max_content_length``, and ``cls``
       parameters.
    """
    return FormDataParser(stream_factory=stream_factory, max_form_memory_size=max_form_memory_size, max_content_length=max_content_length, max_form_parts=max_form_parts, silent=silent, cls=cls).parse_from_environ(environ)