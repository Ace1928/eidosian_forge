from datetime import datetime
from inspect import signature
from io import StringIO
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
def test_escapechar(all_parsers):
    data = 'SEARCH_TERM,ACTUAL_URL\n"bra tv board","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"\n"tv pÃ¥ hjul","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"\n"SLAGBORD, \\"Bergslagen\\", IKEA:s 1700-tals series","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), escapechar='\\', quotechar='"', encoding='utf-8')
    assert result['SEARCH_TERM'][2] == 'SLAGBORD, "Bergslagen", IKEA:s 1700-tals series'
    tm.assert_index_equal(result.columns, Index(['SEARCH_TERM', 'ACTUAL_URL']))