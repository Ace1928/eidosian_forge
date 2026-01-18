from collections import defaultdict
from contextlib import nullcontext
from functools import reduce
import inspect
import json
import os
import re
import operator
import warnings
import pyarrow as pa
from pyarrow._parquet import (ParquetReader, Statistics,  # noqa
from pyarrow.fs import (LocalFileSystem, FileSystem, FileType,
from pyarrow import filesystem as legacyfs
from pyarrow.util import guid, _is_path_like, _stringify_path, _deprecate_api
@property
def num_row_groups(self):
    """
        Return the number of row groups of the Parquet file.

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        >>> parquet_file.num_row_groups
        1
        """
    return self.reader.num_row_groups