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
@pytest.mark.parametrize('version', [114, 117, 118, 119, None])
def test_invalid_timestamp(self, version):
    original = DataFrame([(1,)], columns=['variable'])
    time_stamp = '01 Jan 2000, 00:00:00'
    with tm.ensure_clean() as path:
        msg = 'time_stamp should be datetime type'
        with pytest.raises(ValueError, match=msg):
            original.to_stata(path, time_stamp=time_stamp, version=version)
        assert not os.path.isfile(path)