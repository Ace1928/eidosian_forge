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
def test_attribute_centric_xml():
    pytest.importorskip('lxml')
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<TrainSchedule>\n      <Stations>\n         <station Name="Manhattan" coords="31,460,195,498"/>\n         <station Name="Laraway Road" coords="63,409,194,455"/>\n         <station Name="179th St (Orland Park)" coords="0,364,110,395"/>\n         <station Name="153rd St (Orland Park)" coords="7,333,113,362"/>\n         <station Name="143rd St (Orland Park)" coords="17,297,115,330"/>\n         <station Name="Palos Park" coords="128,281,239,303"/>\n         <station Name="Palos Heights" coords="148,257,283,279"/>\n         <station Name="Worth" coords="170,230,248,255"/>\n         <station Name="Chicago Ridge" coords="70,187,208,214"/>\n         <station Name="Oak Lawn" coords="166,159,266,185"/>\n         <station Name="Ashburn" coords="197,133,336,157"/>\n         <station Name="Wrightwood" coords="219,106,340,133"/>\n         <station Name="Chicago Union Sta" coords="220,0,360,43"/>\n      </Stations>\n</TrainSchedule>'
    df_lxml = read_xml(StringIO(xml), xpath='.//station')
    df_etree = read_xml(StringIO(xml), xpath='.//station', parser='etree')
    df_iter_lx = read_xml_iterparse(xml, iterparse={'station': ['Name', 'coords']})
    df_iter_et = read_xml_iterparse(xml, parser='etree', iterparse={'station': ['Name', 'coords']})
    tm.assert_frame_equal(df_lxml, df_etree)
    tm.assert_frame_equal(df_iter_lx, df_iter_et)