import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
def assert_dataset_fragment_convenience_methods(dataset):
    for fragment in dataset.get_fragments():
        with fragment.open() as nf:
            assert isinstance(nf, pa.NativeFile)
            assert not nf.closed
            assert nf.seekable()
            assert nf.readable()
            assert not nf.writable()