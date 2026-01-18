import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
def test_non_categorical_value_label_name_conversion():
    data = DataFrame({'invalid~!': [1, 1, 2, 3, 5, 8], '6_invalid': [1, 1, 2, 3, 5, 8], 'invalid_name_longer_than_32_characters': [8, 8, 9, 9, 8, 8], 'aggregate': [2, 5, 5, 6, 6, 9], (1, 2): [1, 2, 3, 4, 5, 6]})
    value_labels = {'invalid~!': {1: 'label1', 2: 'label2'}, '6_invalid': {1: 'label1', 2: 'label2'}, 'invalid_name_longer_than_32_characters': {8: 'eight', 9: 'nine'}, 'aggregate': {5: 'five'}, (1, 2): {3: 'three'}}
    expected = {'invalid__': {1: 'label1', 2: 'label2'}, '_6_invalid': {1: 'label1', 2: 'label2'}, 'invalid_name_longer_than_32_char': {8: 'eight', 9: 'nine'}, '_aggregate': {5: 'five'}, '_1__2_': {3: 'three'}}
    with tm.ensure_clean() as path:
        with tm.assert_produces_warning(InvalidColumnName):
            data.to_stata(path, value_labels=value_labels)
        with StataReader(path) as reader:
            reader_value_labels = reader.value_labels()
            assert reader_value_labels == expected