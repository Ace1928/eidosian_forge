import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
def test_cast_to_extension_with_extension_storage():
    array = pa.array([1, 2, 3], pa.int64())
    array.cast(IntegerEmbeddedType())
    array.cast(IntegerType()).cast(IntegerEmbeddedType())