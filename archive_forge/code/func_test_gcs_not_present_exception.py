from io import BytesIO
import os
import pathlib
import tarfile
import zipfile
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
@td.skip_if_installed('gcsfs')
def test_gcs_not_present_exception():
    with tm.external_error_raised(ImportError):
        read_csv('gs://test/test.csv')