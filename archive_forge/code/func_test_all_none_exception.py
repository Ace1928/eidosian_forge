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
def test_all_none_exception(self, version):
    output = [{'none': 'none', 'number': 0}, {'none': None, 'number': 1}]
    output = DataFrame(output)
    output['none'] = None
    with tm.ensure_clean() as path:
        with pytest.raises(ValueError, match='Column `none` cannot be exported'):
            output.to_stata(path, version=version)