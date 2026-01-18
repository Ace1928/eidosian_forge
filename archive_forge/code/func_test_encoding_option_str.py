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
def test_encoding_option_str(xml_baby_names, parser):
    df_file = read_xml(xml_baby_names, parser=parser, encoding='ISO-8859-1').head(5)
    output = df_file.to_xml(encoding='ISO-8859-1', parser=parser)
    if output is not None:
        output = output.replace('<?xml version="1.0" encoding="ISO-8859-1"?', "<?xml version='1.0' encoding='ISO-8859-1'?")
    assert output == encoding_expected