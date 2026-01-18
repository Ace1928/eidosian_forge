from __future__ import annotations
from io import (
import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
def test_index_false_rename_row_root(xml_books, parser):
    expected = "<?xml version='1.0' encoding='utf-8'?>\n<books>\n  <book>\n    <category>cooking</category>\n    <title>Everyday Italian</title>\n    <author>Giada De Laurentiis</author>\n    <year>2005</year>\n    <price>30.0</price>\n  </book>\n  <book>\n    <category>children</category>\n    <title>Harry Potter</title>\n    <author>J K. Rowling</author>\n    <year>2005</year>\n    <price>29.99</price>\n  </book>\n  <book>\n    <category>web</category>\n    <title>Learning XML</title>\n    <author>Erik T. Ray</author>\n    <year>2003</year>\n    <price>39.95</price>\n  </book>\n</books>"
    df_file = read_xml(xml_books, parser=parser)
    with tm.ensure_clean('test.xml') as path:
        df_file.to_xml(path, index=False, root_name='books', row_name='book', parser=parser)
        with open(path, 'rb') as f:
            output = f.read().decode('utf-8').strip()
        output = equalize_decl(output)
        assert output == expected