import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
def test_bad_rc_file():
    """Test bad value raises error."""
    path = os.path.dirname(os.path.abspath(__file__))
    with pytest.raises(ValueError, match='Bad val '):
        read_rcfile(os.path.join(path, '../bad.rcparams'))