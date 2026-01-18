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
def match_types(writer_type, reader_type, named_schemas):
    if isinstance(writer_type, list) or isinstance(reader_type, list):
        return True
    if isinstance(writer_type, dict) or isinstance(reader_type, dict):
        try:
            return match_schemas(writer_type, reader_type, named_schemas)
        except SchemaResolutionError:
            return False
    if writer_type == reader_type:
        return True
    elif writer_type == 'int' and reader_type in ['long', 'float', 'double']:
        return True
    elif writer_type == 'long' and reader_type in ['float', 'double']:
        return True
    elif writer_type == 'float' and reader_type == 'double':
        return True
    elif writer_type == 'string' and reader_type == 'bytes':
        return True
    elif writer_type == 'bytes' and reader_type == 'string':
        return True
    writer_schema = named_schemas['writer'].get(writer_type)
    reader_schema = named_schemas['reader'].get(reader_type)
    if writer_schema is not None and reader_schema is not None:
        return match_types(writer_schema, reader_schema, named_schemas)
    return False