import codecs
import csv
from io import StringIO
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import PY311
from pandas.errors import (
from pandas import DataFrame
import pandas._testing as tm
def test_bad_stream_exception(all_parsers, csv_dir_path):
    path = os.path.join(csv_dir_path, 'sauron.SHIFT_JIS.csv')
    codec = codecs.lookup('utf-8')
    utf8 = codecs.lookup('utf-8')
    parser = all_parsers
    msg = "'utf-8' codec can't decode byte"
    with open(path, 'rb') as handle, codecs.StreamRecoder(handle, utf8.encode, utf8.decode, codec.streamreader, codec.streamwriter) as stream:
        with pytest.raises(UnicodeDecodeError, match=msg):
            parser.read_csv(stream)