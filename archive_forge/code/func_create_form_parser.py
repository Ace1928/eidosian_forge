from __future__ import annotations
import logging
import os
import shutil
import sys
import tempfile
from email.message import Message
from enum import IntEnum
from io import BytesIO
from numbers import Number
from typing import TYPE_CHECKING
from .decoders import Base64Decoder, QuotedPrintableDecoder
from .exceptions import FileError, FormParserError, MultipartParseError, QuerystringParseError
def create_form_parser(headers, on_field, on_file, trust_x_headers=False, config={}):
    """This function is a helper function to aid in creating a FormParser
    instances.  Given a dictionary-like headers object, it will determine
    the correct information needed, instantiate a FormParser with the
    appropriate values and given callbacks, and then return the corresponding
    parser.

    :param headers: A dictionary-like object of HTTP headers.  The only
                    required header is Content-Type.

    :param on_field: Callback to call with each parsed field.

    :param on_file: Callback to call with each parsed file.

    :param trust_x_headers: Whether or not to trust information received from
                            certain X-Headers - for example, the file name from
                            X-File-Name.

    :param config: Configuration variables to pass to the FormParser.
    """
    content_type = headers.get('Content-Type')
    if content_type is None:
        logging.getLogger(__name__).warning('No Content-Type header given')
        raise ValueError('No Content-Type header given!')
    content_type, params = parse_options_header(content_type)
    boundary = params.get(b'boundary')
    content_type = content_type.decode('latin-1')
    file_name = headers.get('X-File-Name')
    form_parser = FormParser(content_type, on_field, on_file, boundary=boundary, file_name=file_name, config=config)
    return form_parser