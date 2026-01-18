import abc
import bz2
from datetime import date, datetime
from decimal import Decimal
import gc
import gzip
import io
import itertools
import os
import select
import shutil
import signal
import string
import tempfile
import threading
import time
import unittest
import weakref
import pytest
import numpy as np
import pyarrow as pa
from pyarrow.csv import (
from pyarrow.tests import util
def test_concatenated(self):
    csv_path = os.path.join(self.tmpdir, self.csv_filename)
    with gzip.open(csv_path, 'wb', 3) as f:
        f.write(b'ab,cd\nef,gh\n')
    with gzip.open(csv_path, 'ab', 3) as f:
        f.write(b'ij,kl\nmn,op\n')
    table = self.read_csv(csv_path)
    assert table.to_pydict() == {'ab': ['ef', 'ij', 'mn'], 'cd': ['gh', 'kl', 'op']}