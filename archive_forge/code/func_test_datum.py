from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_datum(gdb_arrow):
    check_stack_repr(gdb_arrow, 'empty_datum', 'arrow::Datum (empty)')
    check_stack_repr(gdb_arrow, 'scalar_datum', 'arrow::Datum of value arrow::BooleanScalar of null value')
    check_stack_repr(gdb_arrow, 'array_datum', re.compile('^arrow::Datum of value arrow::ArrayData of type '))
    check_stack_repr(gdb_arrow, 'chunked_array_datum', re.compile('^arrow::Datum of value arrow::ChunkedArray of type '))
    check_stack_repr(gdb_arrow, 'batch_datum', re.compile('^arrow::Datum of value arrow::RecordBatch with 2 columns, 3 rows '))
    check_stack_repr(gdb_arrow, 'table_datum', re.compile('^arrow::Datum of value arrow::Table with 2 columns, 5 rows '))