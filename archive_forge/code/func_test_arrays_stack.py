from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_arrays_stack(gdb_arrow):
    check_stack_repr(gdb_arrow, 'int32_array', 'arrow::Int32Array of length 4, offset 0, null count 1 = {[0] = -5, [1] = 6, [2] = null, [3] = 42}')
    check_stack_repr(gdb_arrow, 'list_array', 'arrow::ListArray of type arrow::list(arrow::int64()), length 3, offset 0, null count 1')