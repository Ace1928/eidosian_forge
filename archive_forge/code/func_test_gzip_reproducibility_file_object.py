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
def test_gzip_reproducibility_file_object():
    """
    Gzip should create reproducible archives with mtime.

    GH 28103
    """
    df = pd.DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD'), dtype=object), index=pd.Index([f'i-{i}' for i in range(30)], dtype=object))
    compression_options = {'method': 'gzip', 'mtime': 1}
    buffer = io.BytesIO()
    df.to_csv(buffer, compression=compression_options, mode='wb')
    output = buffer.getvalue()
    time.sleep(0.1)
    buffer = io.BytesIO()
    df.to_csv(buffer, compression=compression_options, mode='wb')
    assert output == buffer.getvalue()