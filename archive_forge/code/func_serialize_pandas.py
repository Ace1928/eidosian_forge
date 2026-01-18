import os
import pyarrow as pa
from pyarrow.lib import (IpcReadOptions, IpcWriteOptions, ReadStats, WriteStats,  # noqa
import pyarrow.lib as lib
def serialize_pandas(df, *, nthreads=None, preserve_index=None):
    """
    Serialize a pandas DataFrame into a buffer protocol compatible object.

    Parameters
    ----------
    df : pandas.DataFrame
    nthreads : int, default None
        Number of threads to use for conversion to Arrow, default all CPUs.
    preserve_index : bool, default None
        The default of None will store the index as a column, except for
        RangeIndex which is stored as metadata only. If True, always
        preserve the pandas index data as a column. If False, no index
        information is saved and the result will have a default RangeIndex.

    Returns
    -------
    buf : buffer
        An object compatible with the buffer protocol.
    """
    batch = pa.RecordBatch.from_pandas(df, nthreads=nthreads, preserve_index=preserve_index)
    sink = pa.BufferOutputStream()
    with pa.RecordBatchStreamWriter(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()