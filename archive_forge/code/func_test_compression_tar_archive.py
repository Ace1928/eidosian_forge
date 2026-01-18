import os
from pathlib import Path
import tarfile
import zipfile
import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_compression_tar_archive(all_parsers, csv_dir_path):
    parser = all_parsers
    path = os.path.join(csv_dir_path, 'tar_csv.tar.gz')
    df = parser.read_csv(path)
    assert list(df.columns) == ['a']