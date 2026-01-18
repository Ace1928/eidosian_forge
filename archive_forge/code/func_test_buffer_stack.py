from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_buffer_stack(gdb_arrow):
    check_stack_repr(gdb_arrow, 'buffer_null', 'arrow::Buffer of size 0, read-only')
    check_stack_repr(gdb_arrow, 'buffer_abc', 'arrow::Buffer of size 3, read-only, "abc"')
    check_stack_repr(gdb_arrow, 'buffer_special_chars', 'arrow::Buffer of size 12, read-only, "foo\\"bar\\000\\r\\n\\t\\037"')
    check_stack_repr(gdb_arrow, 'buffer_mutable', 'arrow::MutableBuffer of size 3, mutable, "abc"')