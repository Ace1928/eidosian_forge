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
def read_row_groups(self, row_groups, columns=None, use_threads=True, use_pandas_metadata=False):
    """
        Read a multiple row groups from a Parquet file.

        Parameters
        ----------
        row_groups : list
            Only these row groups will be read from the file.
        columns : list
            If not None, only these columns will be read from the row group. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.table.Table
            Content of the row groups as a table (of columns).

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        >>> parquet_file.read_row_groups([0,0])
        pyarrow.Table
        n_legs: int64
        animal: string
        ----
        n_legs: [[2,2,4,4,5,...,2,4,4,5,100]]
        animal: [["Flamingo","Parrot","Dog",...,"Brittle stars","Centipede"]]
        """
    column_indices = self._get_column_indices(columns, use_pandas_metadata=use_pandas_metadata)
    return self.reader.read_row_groups(row_groups, column_indices=column_indices, use_threads=use_threads)