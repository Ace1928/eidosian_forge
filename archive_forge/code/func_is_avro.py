import bz2
import json
import lzma
import zlib
from datetime import datetime, timezone
from decimal import Context
from io import BytesIO
from struct import error as StructError
from typing import IO, Union, Optional, Generic, TypeVar, Iterator, Dict
from warnings import warn
from .io.binary_decoder import BinaryDecoder
from .io.json_decoder import AvroJSONDecoder
from .logical_readers import LOGICAL_READERS
from .schema import (
from .types import Schema, AvroMessage, NamedSchemas
from ._read_common import (
from .const import NAMED_TYPES, AVRO_TYPES
def is_avro(path_or_buffer: Union[str, IO]) -> bool:
    """Return True if path (or buffer) points to an Avro file. This will only
    work for avro files that contain the normal avro schema header like those
    create from :func:`~fastavro._write_py.writer`. This function is not intended
    to be used with binary data created from
    :func:`~fastavro._write_py.schemaless_writer` since that does not include the
    avro header.

    Parameters
    ----------
    path_or_buffer
        Path to file
    """
    fp: IO
    if isinstance(path_or_buffer, str):
        fp = open(path_or_buffer, 'rb')
        close = True
    else:
        fp = path_or_buffer
        close = False
    try:
        header = fp.read(len(MAGIC))
        return header == MAGIC
    finally:
        if close:
            fp.close()