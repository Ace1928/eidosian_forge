import os
import random
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _test_dataframe
from pyarrow.tests.parquet.test_dataset import (
from pyarrow.util import guid
def test_capacity_space(self):
    capacity = self.hdfs.get_capacity()
    space_used = self.hdfs.get_space_used()
    disk_free = self.hdfs.df()
    assert capacity > 0
    assert capacity > space_used
    assert disk_free == capacity - space_used