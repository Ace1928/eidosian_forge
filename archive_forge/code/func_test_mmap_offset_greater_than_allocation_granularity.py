import sys
import os
import mmap
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from numpy import (
from numpy import arange, allclose, asarray
from numpy.testing import (
def test_mmap_offset_greater_than_allocation_granularity(self):
    size = 5 * mmap.ALLOCATIONGRANULARITY
    offset = mmap.ALLOCATIONGRANULARITY + 1
    fp = memmap(self.tmpfp, shape=size, mode='w+', offset=offset)
    assert_(fp.offset == offset)