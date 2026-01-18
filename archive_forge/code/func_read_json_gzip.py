import dataclasses
import datetime
import gzip
import json
import numbers
import pathlib
from typing import (
import numpy as np
import pandas as pd
import sympy
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def read_json_gzip(file_or_fn: Union[None, IO, pathlib.Path, str]=None, *, gzip_raw: Optional[bytes]=None, resolvers: Optional[Sequence[JsonResolver]]=None):
    """Read a gzipped JSON file that optionally contains cirq objects.

    Args:
        file_or_fn: A filename (if a string or `pathlib.Path`) to read from, or
            an IO object (such as a file or buffer) to read from, or `None` to
            indicate that `gzip_raw` argument should be used. Defaults to
            `None`.
        gzip_raw: Bytes representing the raw gzip input to unzip and parse
            or else `None` indicating `file_or_fn` should be used. Defaults to
            `None`.
        resolvers: A list of functions that are called in order to turn
            the serialized `cirq_type` string into a constructable class.
            By default, top-level cirq objects that implement the SupportsJSON
            protocol are supported. You can extend the list of supported types
            by pre-pending custom resolvers. Each resolver should return `None`
            to indicate that it cannot resolve the given cirq_type and that
            the next resolver should be tried.

    Raises:
        ValueError: If either none of `file_or_fn` and `gzip_raw` is specified,
            or both are specified.
    """
    if (file_or_fn is None) == (gzip_raw is None):
        raise ValueError('Must specify ONE of "file_or_fn" or "gzip_raw".')
    if gzip_raw is not None:
        json_str = gzip.decompress(gzip_raw).decode(encoding='utf-8')
        return read_json(json_text=json_str, resolvers=resolvers)
    with gzip.open(file_or_fn, 'rt') as json_file:
        return read_json(cast(IO, json_file), resolvers=resolvers)