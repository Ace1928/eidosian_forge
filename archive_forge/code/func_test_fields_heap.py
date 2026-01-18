from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_fields_heap(gdb_arrow):
    check_heap_repr(gdb_arrow, 'heap_int_field', 'arrow::field("ints", arrow::int64())')