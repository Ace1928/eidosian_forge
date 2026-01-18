from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_chunked_array(gdb_arrow):
    check_stack_repr(gdb_arrow, 'chunked_array', 'arrow::ChunkedArray of type arrow::int32(), length 5, null count 1 with 2 chunks = {[0] = length 2, offset 0, null count 0, [1] = length 3, offset 0, null count 1}')