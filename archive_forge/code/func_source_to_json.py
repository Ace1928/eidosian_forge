from __future__ import annotations
import pathlib
from typing import IO, TYPE_CHECKING, Any, Optional, TextIO, Tuple, Union
from io import TextIOBase, TextIOWrapper
from posixpath import normpath, sep
from urllib.parse import urljoin, urlsplit, urlunsplit
from rdflib.parser import (
def source_to_json(source: Optional[Union[IO[bytes], TextIO, InputSource, str, bytes, pathlib.PurePath]]) -> Optional[Any]:
    if isinstance(source, PythonInputSource):
        return source.data
    if isinstance(source, StringInputSource):
        return json.load(source.getCharacterStream())
    source = create_input_source(source, format='json-ld')
    stream = source.getByteStream()
    try:
        if isinstance(stream, BytesIOWrapper):
            stream = stream.wrapped
        if isinstance(stream, TextIOBase):
            use_stream = stream
        else:
            use_stream = TextIOWrapper(stream, encoding='utf-8')
        return json.load(use_stream)
    finally:
        stream.close()