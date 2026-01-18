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
def test_dtd(parser):
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE non-profits [\n    <!ELEMENT shapes (shape*) >\n    <!ELEMENT shape ( name, type )>\n    <!ELEMENT name (#PCDATA)>\n]>\n<shapes>\n  <shape>\n    <name>circle</name>\n    <type>2D</type>\n  </shape>\n  <shape>\n    <name>sphere</name>\n    <type>3D</type>\n  </shape>\n</shapes>'
    df_xpath = read_xml(StringIO(xml), xpath='.//shape', parser=parser)
    df_iter = read_xml_iterparse(xml, parser=parser, iterparse={'shape': ['name', 'type']})
    df_expected = DataFrame({'name': ['circle', 'sphere'], 'type': ['2D', '3D']})
    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)