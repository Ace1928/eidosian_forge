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
@pytest.fixture
def from_file_expected():
    return "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <index>0</index>\n    <category>cooking</category>\n    <title>Everyday Italian</title>\n    <author>Giada De Laurentiis</author>\n    <year>2005</year>\n    <price>30.0</price>\n  </row>\n  <row>\n    <index>1</index>\n    <category>children</category>\n    <title>Harry Potter</title>\n    <author>J K. Rowling</author>\n    <year>2005</year>\n    <price>29.99</price>\n  </row>\n  <row>\n    <index>2</index>\n    <category>web</category>\n    <title>Learning XML</title>\n    <author>Erik T. Ray</author>\n    <year>2003</year>\n    <price>39.95</price>\n  </row>\n</data>"