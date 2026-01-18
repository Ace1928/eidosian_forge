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
@pytest.mark.parametrize('write_method, write_kwargs, read_method', [('to_csv', {'index': False}, pd.read_csv), ('to_json', {}, pd.read_json), ('to_pickle', {}, pd.read_pickle)])
def test_dataframe_compression_defaults_to_infer(write_method, write_kwargs, read_method, compression_only, compression_to_extension):
    input = pd.DataFrame([[1.0, 0, -4], [3.4, 5, 2]], columns=['X', 'Y', 'Z'])
    extension = compression_to_extension[compression_only]
    with tm.ensure_clean('compressed' + extension) as path:
        getattr(input, write_method)(path, **write_kwargs)
        output = read_method(path, compression=compression_only)
    tm.assert_frame_equal(output, input)