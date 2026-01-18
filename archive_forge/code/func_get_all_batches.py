import io
import os
import sys
import pytest
import pyarrow as pa
def get_all_batches(f):
    for row_group in range(f.num_row_groups):
        batches = f.iter_batches(batch_size=900, row_groups=[row_group])
        for batch in batches:
            yield batch