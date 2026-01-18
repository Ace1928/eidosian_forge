import gzip
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap
import time
import zipfile
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom
@pytest.mark.parametrize('obj', [pd.DataFrame(100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]], columns=['X', 'Y', 'Z']), pd.Series(100 * [0.123456, 0.234567, 0.567567], name='X')])
@pytest.mark.parametrize('method', ['to_pickle', 'to_json', 'to_csv'])
def test_gzip_compression_level(obj, method):
    with tm.ensure_clean() as path:
        getattr(obj, method)(path, compression='gzip')
        compressed_size_default = os.path.getsize(path)
        getattr(obj, method)(path, compression={'method': 'gzip', 'compresslevel': 1})
        compressed_size_fast = os.path.getsize(path)
        assert compressed_size_default < compressed_size_fast