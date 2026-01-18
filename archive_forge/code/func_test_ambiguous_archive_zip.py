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
def test_ambiguous_archive_zip():
    with tm.ensure_clean(filename='.zip') as path:
        with zipfile.ZipFile(path, 'w') as file:
            file.writestr('a.csv', 'foo,bar')
            file.writestr('b.csv', 'foo,bar')
        with pytest.raises(ValueError, match='Multiple files found in ZIP file'):
            pd.read_csv(path)