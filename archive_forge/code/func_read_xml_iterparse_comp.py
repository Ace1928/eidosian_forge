from __future__ import annotations
from io import (
from lzma import LZMAError
import os
from tarfile import ReadError
from urllib.error import HTTPError
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
def read_xml_iterparse_comp(comp_path, compression_only, **kwargs):
    with get_handle(comp_path, 'r', compression=compression_only) as handles:
        with tm.ensure_clean() as path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(handles.handle.read())
            return read_xml(path, **kwargs)