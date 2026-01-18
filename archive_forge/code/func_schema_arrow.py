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
def schema_arrow(self):
    """
        Return the inferred Arrow schema, converted from the whole Parquet
        file's schema

        Examples
        --------
        Generate an example Parquet file:

        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        Read the Arrow schema:

        >>> parquet_file.schema_arrow
        n_legs: int64
        animal: string
        """
    return self.reader.schema_arrow