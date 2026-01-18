from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_array_data(gdb_arrow):
    check_stack_repr(gdb_arrow, 'int32_array_data', 'arrow::ArrayData of type arrow::int32(), length 4, offset 0, null count 1 = {[0] = -5, [1] = 6, [2] = null, [3] = 42}')