import sys
import os
import mmap
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from numpy import (
from numpy import arange, allclose, asarray
from numpy.testing import (
def test_filename_fileobj(self):
    fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+', shape=self.shape)
    assert_equal(fp.filename, self.tmpfp.name)