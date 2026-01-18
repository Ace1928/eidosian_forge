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
def test_ipc_registered():
    with registered_extension_type(ParamExtType(1)):
        batch = example_batch()
        buf = ipc_write_batch(batch)
        del batch
        batch = ipc_read_batch(buf)
        batch.validate(full=True)
        arr = check_example_batch(batch, expect_extension=True)
        assert arr.type == ParamExtType(3)