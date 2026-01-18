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
def test_file_only_attrs(xml_books, parser):
    df_file = read_xml(xml_books, attrs_only=True, parser=parser)
    df_iter = read_xml(xml_books, parser=parser, iterparse={'book': ['category']})
    df_expected = DataFrame({'category': ['cooking', 'children', 'web']})
    tm.assert_frame_equal(df_file, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)