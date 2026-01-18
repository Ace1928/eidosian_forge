from __future__ import annotations
import inspect
import pathlib
import pickle
from typing import IO, AnyStr, Callable, Iterator, Literal, Optional, Union
import pandas
import pandas._libs.lib as lib
from pandas._typing import CompressionOptions, DtypeArg, DtypeBackend, StorageOptions
from modin.core.storage_formats import BaseQueryCompiler
from modin.utils import expanduser_path_arg
from . import DataFrame
@expanduser_path_arg('path_or_buffer')
def read_xml_glob(path_or_buffer, *, xpath='./*', namespaces=None, elems_only=False, attrs_only=False, names=None, dtype=None, converters=None, parse_dates=None, encoding='utf-8', parser='lxml', stylesheet=None, iterparse=None, compression='infer', storage_options: StorageOptions=None, dtype_backend=lib.no_default) -> DataFrame:
    """
    Read XML document into a DataFrame object.

    This experimental feature provides parallel reading from multiple XML files which are
    defined by glob pattern. The files must contain parts of one dataframe, which can be
    obtained, for example, by `DataFrame.modin.to_xml_glob` function.

    Returns
    -------
    DataFrame

    Notes
    -----
    * Only string type supported for `path_or_buffer` argument.
    * The rest of the arguments are the same as for `pandas.read_xml`.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    return DataFrame(query_compiler=FactoryDispatcher.read_xml_glob(path_or_buffer=path_or_buffer, xpath=xpath, namespaces=namespaces, elems_only=elems_only, attrs_only=attrs_only, names=names, dtype=dtype, converters=converters, parse_dates=parse_dates, encoding=encoding, parser=parser, stylesheet=stylesheet, iterparse=iterparse, compression=compression, storage_options=storage_options, dtype_backend=dtype_backend))