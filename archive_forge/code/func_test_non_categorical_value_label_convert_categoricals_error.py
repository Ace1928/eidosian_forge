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
def test_non_categorical_value_label_convert_categoricals_error():
    value_labels = {'repeated_labels': {10: 'Ten', 20: 'More than ten', 40: 'More than ten'}}
    data = DataFrame({'repeated_labels': [10, 10, 20, 20, 40, 40]})
    with tm.ensure_clean() as path:
        data.to_stata(path, value_labels=value_labels)
        with StataReader(path, convert_categoricals=False) as reader:
            reader_value_labels = reader.value_labels()
        assert reader_value_labels == value_labels
        col = 'repeated_labels'
        repeats = '-' * 80 + '\n' + '\n'.join(['More than ten'])
        msg = f'\nValue labels for column {col} are not unique. These cannot be converted to\npandas categoricals.\n\nEither read the file with `convert_categoricals` set to False or use the\nlow level interface in `StataReader` to separately read the values and the\nvalue_labels.\n\nThe repeated labels are:\n{repeats}\n'
        with pytest.raises(ValueError, match=msg):
            read_stata(path, convert_categoricals=True)