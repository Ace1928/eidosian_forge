from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_buffer_heap(gdb_arrow):
    check_heap_repr(gdb_arrow, 'heap_buffer', 'arrow::Buffer of size 3, read-only, "abc"')
    check_heap_repr(gdb_arrow, 'heap_buffer_mutable.get()', 'arrow::Buffer of size 3, mutable, "abc"')